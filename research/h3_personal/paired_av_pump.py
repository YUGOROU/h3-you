"""Foreground lifetime pumps over caller-authenticated sockets, no connect/listen.

At most two sender reservations. An owned ACK reader overlaps native generation
with receiver pipe writes; it is joined/reaped on exit, not a provider watcher.
No retry after partial IO, uncertain ACK or failed sink. Model cleanup/health is
separate: pumps never start, stop or restart H3 or access OBS/devices themselves.
"""
from pathlib import Path
from collections import deque
import os
import socket
import threading
import time

from .paired_av import Receiver,ProtocolError,LiveAVBinding
from .paired_av_sender import Sender
from . import paired_av_wire as wire

MAX_PAIR=16*1024*1024+124*4000


def _close(sock):
    try:sock.shutdown(socket.SHUT_RDWR)
    except OSError:pass
    sock.close()


class ProducerPump:
    def __init__(self,sock,binding,*,deadline,overlap_send=False):
        if type(binding) is not LiveAVBinding:raise ProtocolError('explicit live binding required')
        if binding.max_inflight_bytes<2*MAX_PAIR:
            raise ValueError('two worst-case pair credits required for overlapping producer')
        self.sock=sock;self.sender=Sender(binding,deadline);self.deadline=deadline
        self._cv=threading.Condition();self._error=None;self._stop=False;self._thread=None;self._ran=False
        if type(overlap_send) is not bool:raise ValueError('explicit overlap_send bool required')
        self.overlap_send=overlap_send
        self._send_queue=deque();self._send_thread=None
        self.highwater=0
        self._origin=time.monotonic();self._marks={}

    def _mark(self,index,stage,**values):
        # Called with _cv held; no payload or cross-process timestamps.
        if index not in self._marks and len(self._marks)>=self.sender.binding.max_segments:return
        row=self._marks.setdefault(index,{'index':index})
        row[stage]=round(time.monotonic()-self._origin,6)
        row.update(values)

    def timing(self):
        with self._cv:
            return dict(clock='local_monotonic',origin_monotonic=self._origin,
                        send_semantics='local_enqueue_not_peer_delivery',
                        windows=[dict(self._marks[k]) for k in sorted(self._marks)])

    def _check(self):
        if self._error is not None:raise RuntimeError('paired transport failed; no replay') from self._error
        wire._remaining(self.deadline)

    def _acks(self):
        try:
            while True:
                with self._cv:
                    if self._stop:return
                msg=wire.receive_ack_message(self.sock,deadline=self.deadline)
                with self._cv:
                    self.sender.ack(msg)
                    self._mark(msg['index'], 'ack_'+msg['stage'].replace('-','_'))
                    state=self.sender.snapshot();self._cv.notify_all()
                    if state['final'] and not state['pending']:return
        except BaseException as exc:
            with self._cv:
                if not self._stop:self._error=exc;self.sender.disconnect()
                self._cv.notify_all()
            try:self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass

    def _send(self,packet,ticket):
        index=packet.envelope.index
        with self._cv:
            self._check();self._mark(index,'send_started')
        try:wire.send_pair(self.sock,packet.envelope,packet.video,packet.pcm,deadline=self.deadline)
        except BaseException:
            with self._cv:self.sender.failed(ticket)
            raise
        with self._cv:self._mark(index,'send_enqueued')

    def _send_loop(self):
        try:
            while True:
                with self._cv:
                    while not self._send_queue and not self._stop:
                        self._cv.wait(min(.05,wire._remaining(self.deadline)))
                    if self._stop:return
                    packet,ticket=self._send_queue.popleft()
                self._send(packet,ticket)
        except BaseException as exc:
            with self._cv:
                if not self._stop:self._error=exc;self.sender.disconnect()
                self._cv.notify_all()
            try:self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass

    def run(self,packets):
        """One generator owner; queued/sending/unacked pairs share two credits.

        Optional sender thread overlaps blocking local enqueue with next pull.
        Generation never moves to a background thread or outlives this call.
        """
        if self._ran:raise RuntimeError('pump cannot be replayed')
        self._ran=True
        self._thread=threading.Thread(target=self._acks,name='h3-live-av-acks')
        started=False;send_started=False;iterator=None
        try:
            self._thread.start();started=True
            if self.overlap_send:
                self._send_thread=threading.Thread(target=self._send_loop,name='h3-live-av-send')
                self._send_thread.start();send_started=True
            iterator=iter(packets)
            index=0
            while True:
                with self._cv:
                    self._check()
                    self._mark(index,'credit_wait_started')
                    while self.sender.snapshot()['pending']>=2:
                        self._cv.wait(min(.05,wire._remaining(self.deadline)));self._check()
                    self._mark(index,'pull_started')
                try:packet=next(iterator)
                except StopIteration:break
                with self._cv:
                    self._mark(index,'packet_ready',video_bytes=len(packet.video),pcm_bytes=len(packet.pcm))
                    self._check();ticket=self.sender.reserve(packet.envelope)
                    self.highwater=max(self.highwater,self.sender.snapshot()['pending'])
                    if self.overlap_send:
                        self._send_queue.append((packet,ticket));self._cv.notify_all()
                if not self.overlap_send:self._send(packet,ticket)
                if packet.envelope.conversation_final:break
                index+=1
            with self._cv:
                if not self.sender.snapshot()['final']:raise ProtocolError('producer ended without explicit final pair')
                while self.sender.snapshot()['pending']:
                    self._check();self._cv.wait(min(.05,wire._remaining(self.deadline)))
                self._check()
                return dict(self.sender.snapshot(),queue_highwater=self.highwater)
        except BaseException:
            with self._cv:self.sender.disconnect()
            if hasattr(self.sock,'abort'):self.sock.abort()
            raise
        finally:
            with self._cv:self._stop=True;self._cv.notify_all()
            _close(self.sock)
            alive=[]
            reap_limit=time.monotonic()+5
            for active,thread,label in ((started,self._thread,'ACK reader'),
                                        (send_started,self._send_thread,'send writer')):
                if active:
                    thread.join(timeout=max(0.1, min(2.0, reap_limit-time.monotonic())))
                    if thread.is_alive():alive.append(label)
            with self._cv:self._send_queue.clear()
            if iterator is not None and hasattr(iterator,'close'):iterator.close()
            if alive:raise RuntimeError(', '.join(alive)+' did not terminate')


class ReceiverPump:
    def __init__(self,sock,binding,root,sink,*,deadline,prefetch=False):
        self.root=Path(root).resolve(strict=True)
        if (self.root.is_relative_to(Path(__file__).resolve().parents[2])
                or self.root.stat().st_mode & 0o077 or self.root!=sink.root or sink.binding!=binding):
            raise ValueError('private external sink root/binding required')
        self.sock=sock;self.receiver=Receiver(binding,deadline=deadline);self.sink=sink;self.deadline=deadline
        if type(prefetch) is not bool:raise ValueError('explicit prefetch bool required')
        self.prefetch=prefetch;self._ran=False
        self._cv=threading.Condition();self._ack_lock=threading.Lock()
        self._receipts=[];self._error=None;self._stop=False;self._thread=None
        self._close_lock=threading.Lock();self._closed=False
        self._origin=time.monotonic();self._marks={}

    def _mark(self,index,stage):
        """Bounded local timing only: no payload, no cross-process clock claim."""
        with self._cv:
            if len(self._marks)>self.receiver.binding.max_segments:return
            entry=self._marks.setdefault(index,{'index':index})
            entry[stage]=round(time.monotonic()-self._origin,4)

    def timing(self):
        with self._cv:
            return self._timing_locked()

    def _timing_locked(self):
        return dict(receive=[dict(self._marks[k]) for k in sorted(self._marks)],
                    prefetch=self.prefetch, clock='local_monotonic',origin_monotonic=self._origin)

    def _close_transport(self):
        with self._close_lock:
            if self._closed:return
            self._closed=True
            try:
                if hasattr(self.sock,'abort'):self.sock.abort()
            finally:
                # Keep the descriptor valid until the reader exits: closing while
                # recv is entering can race descriptor use on some platforms.
                try:self.sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass

    def cancel(self):
        """Wake transport waits; a caller-owned blocking sink must honor cancel."""
        with self._cv:
            self._stop=True;self.receiver.cancel();self._cv.notify_all()
        self._close_transport()
        self.sink.cancel()

    def _receive_ahead(self):
        try:
            while True:
                with self._cv:
                    while True:
                        if self._stop:return
                        state=self.receiver.snapshot()
                        if state['terminal'] is not None:raise ProtocolError(state['terminal'])
                        if not self._receipts and state['pending']+state['leased']<2:break
                        self._cv.wait(wire._remaining(self.deadline))
                receipt=wire.receive_pair(self.sock,self.receiver,deadline=self.deadline,state_lock=self._cv)
                self._mark(receipt.envelope.index,'received')
                with self._ack_lock:
                    wire.send_ack(self.sock,receipt.envelope,stage='verified',deadline=self.deadline)
                self._mark(receipt.envelope.index,'verified')
                with self._cv:
                    if self._stop:return
                    self._receipts.append(receipt);self._cv.notify_all()
                if receipt.envelope.conversation_final:return
        except BaseException as exc:
            with self._cv:
                if not self._stop:self._error=exc
                self.receiver.disconnect();self._cv.notify_all()
            self._close_transport()

    def _submit(self, path, lease):
        """Explicit epoch sink opt-in; old sinks retain their original behavior."""
        from .epoch_av_sink import PresentationInvalidated
        predicate = getattr(self.sink, 'should_discard', None)
        discard = False if predicate is None else predicate(lease)
        if type(discard) is not bool:
            raise ProtocolError('explicit presentation disposition required')
        cut=getattr(lease.segment, 'presentation_cut', None)
        if cut is not None and getattr(self.sink, 'supports_paired_cuts', False) is not True:
            raise ProtocolError('explicit paired cut sink required')
        archive=getattr(self.sink, 'review_archive', None)
        if archive is not None and cut is not None:
            archive.capture_pair(lease.segment, lease.video, lease.audio_pcm)
        if not discard:
            try:
                self.sink.accept(path.name, lease)
            except PresentationInvalidated:
                discard = True
        if discard:
            self.sink.discard(lease)
        self._mark(lease.segment.index, 'discarded_interrupted' if discard else
                   'partially_presented_interrupted' if cut is not None else 'accepted')
        return discard

    def _ack_disposition(self, lease, discarded):
        with self._cv:
            if self._error is not None:raise self._error
            if self._stop:raise ProtocolError('receiver cancelled')
            partial=not discarded and getattr(lease.segment, 'presentation_cut', None) is not None
            if discarded:self.receiver.discard_interrupted(lease, certain=True)
            elif partial:self.receiver.ack_partial(lease, certain=True)
            else:self.receiver.ack(lease,certain=True)
            self._cv.notify_all()
        stage=('discarded-interrupted' if discarded else
               'partially-presented-interrupted' if partial else 'sink-accepted')
        with self._ack_lock:
            wire.send_ack(self.sock,lease.segment,stage=stage,deadline=self.deadline)
        self._mark(lease.segment.index, 'discard_acked' if discarded else
                   'partial_acked' if partial else 'sink_acked')

    def _run_prefetch(self):
        owned=[];output=None;primary=None;started=False
        self._thread=threading.Thread(target=self._receive_ahead,name='h3-live-av-receive')
        try:
            self._thread.start();started=True
            while True:
                with self._cv:
                    while not self._receipts:
                        if self._error is not None:raise self._error
                        if self._stop:raise ProtocolError('receiver cancelled')
                        self._cv.wait(wire._remaining(self.deadline))
                    if self._error is not None:raise self._error
                    receipt=self._receipts.pop(0);lease=self.receiver.lease();self._cv.notify_all()
                e=receipt.envelope
                path=self.root/f'wire-{e.index:03d}.mp4'
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);owned.append(path)
                with os.fdopen(fd,'wb') as stream:stream.write(lease.video)
                path.chmod(0o400)
                self._mark(e.index,'accept_start')
                discarded=self._submit(path,lease)
                output=self.sink.finish() if e.conversation_final else None
                self._ack_disposition(lease,discarded)
                path.unlink();owned.remove(path)
                if e.conversation_final:
                    with self._cv:return dict(self.receiver.snapshot(),output=output,timing=self._timing_locked())
        except BaseException as exc:
            primary=exc
            with self._cv:self.receiver.disconnect()
            try:self.sink.cancel()
            except BaseException:pass
            raise
        finally:
            with self._cv:
                self._stop=True;self._receipts.clear();self._cv.notify_all()
            try:
                self._close_transport()
            except BaseException:
                if primary is None:raise
            finally:
                if started:self._thread.join(timeout=2)
                self.sock.close()
                for path in owned:path.unlink(missing_ok=True)
            if started and self._thread.is_alive() and primary is None:
                raise RuntimeError('receive reader did not terminate')

    def run(self):
        if self._ran:raise RuntimeError('pump cannot be replayed')
        self._ran=True
        if self.prefetch:return self._run_prefetch()
        owned=[]
        try:
            while not self.receiver.snapshot()['complete']:
                receipt=wire.receive_pair(self.sock,self.receiver,deadline=self.deadline)
                e=receipt.envelope
                self._mark(e.index,'received')
                wire.send_ack(self.sock,e,stage='verified',deadline=self.deadline)
                self._mark(e.index,'verified')
                lease=self.receiver.lease()
                path=self.root/f'wire-{e.index:03d}.mp4'
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);owned.append(path)
                with os.fdopen(fd,'wb') as stream:stream.write(lease.video)
                path.chmod(0o400)
                self._mark(e.index,'accept_start')
                discarded=self._submit(path,lease)
                output=self.sink.finish() if e.conversation_final else None
                self._ack_disposition(lease,discarded)
                path.unlink();owned.remove(path)
            return dict(self.receiver.snapshot(),output=output,timing=self.timing())
        except BaseException:
            self.receiver.disconnect();self.sink.cancel()
            if hasattr(self.sock,'abort'):self.sock.abort()
            raise
        finally:
            _close(self.sock)
            for path in owned:path.unlink(missing_ok=True)
