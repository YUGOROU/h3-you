import importlib.util
import contextlib
import io
import json
import time
import unittest
from unittest.mock import patch
from research.h3_personal.webrtc_ice import HOST_ONLY,CLOUDFLARE_STUN,STUN_URL,HF_CLOUDFLARE_TURN_TLS,TURN_TLS_URL,ice_urls


HAVE_RTC = importlib.util.find_spec('av') is not None and importlib.util.find_spec('aiortc') is not None


class ICEProfiles(unittest.TestCase):
    def test_default_remains_host_only_and_allowlist_exact(self):
        self.assertEqual(ice_urls(),[])
        self.assertEqual(ice_urls(CLOUDFLARE_STUN),[STUN_URL])
        self.assertEqual(ice_urls(HF_CLOUDFLARE_TURN_TLS),[TURN_TLS_URL])
        for bad in ('turn:turn.cloudflare.com:3478','stun:other.invalid:3478','',None,[],True):
            with self.subTest(bad=bad),self.assertRaises(ValueError):ice_urls(bad)

    def test_offline_opt_in_does_not_launch(self):
        from research import run_h3_webrtc_connect as owner
        with patch.object(owner,'run') as run,contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(owner.main(['--cloudflare-stun']),0)
        run.assert_not_called();row=json.loads(output.getvalue())
        self.assertEqual(row['external_ice_servers'],[STUN_URL])
        self.assertEqual(row['scope'],'offline')


    def test_turn_offline_does_not_request_credentials(self):
        from research import run_h3_webrtc_connect as owner
        with patch.object(owner,'run') as run,patch.object(owner,'fetch_turn_credentials') as fetch,contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(owner.main(['--hf-turn']),0)
        run.assert_not_called();fetch.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())['external_ice_servers'],[TURN_TLS_URL])


@unittest.skipUnless(HAVE_RTC, 'isolated locked uv WebRTC environment required')
class PeerConfiguration(unittest.IsolatedAsyncioTestCase):
    async def test_turn_explicit_credentials_without_gather(self):
        from research.h3_personal import webrtc_live_peer as peer
        config=dict(urls=TURN_TLS_URL,username='test-user',credential='test-password')
        with patch.object(peer,'RTCPeerConnection',wraps=peer.RTCPeerConnection) as create:
            async with peer.Session(deadline=time.monotonic()+5,ice_profile=HF_CLOUDFLARE_TURN_TLS,turn_config=config) as session:
                server=create.call_args.args[0].iceServers[0]
                self.assertEqual(server.urls,TURN_TLS_URL)
                self.assertEqual(server.username,'test-user')
                self.assertEqual(server.credential,'test-password')
                self.assertEqual(session.pc.iceGatheringState,'new')
        for profile,turn in ((HF_CLOUDFLARE_TURN_TLS,None),(HOST_ONLY,config),(CLOUDFLARE_STUN,config)):
            with self.subTest(profile=profile),patch.object(peer,'RTCPeerConnection') as allocate,self.assertRaises(ValueError):
                peer.Session(deadline=time.monotonic()+5,ice_profile=profile,turn_config=turn)
            allocate.assert_not_called()

    async def test_stun_configuration_no_gather_or_network(self):
        from research.h3_personal import webrtc_live_peer as peer
        original=peer.RTCPeerConnection
        with patch.object(peer,'RTCPeerConnection',wraps=original) as create:
            async with peer.Session(deadline=time.monotonic()+5,ice_profile=CLOUDFLARE_STUN) as session:
                config=create.call_args.args[0]
                self.assertEqual(len(config.iceServers),1)
                server=config.iceServers[0]
                self.assertEqual(server.urls,STUN_URL)
                self.assertIsNone(server.username);self.assertIsNone(server.credential)
                self.assertEqual(session.pc.iceGatheringState,'new')
                self.assertEqual(session.ice_profile,CLOUDFLARE_STUN)

    async def test_invalid_profile_before_peer_allocation(self):
        from research.h3_personal import webrtc_live_peer as peer
        with patch.object(peer,'RTCPeerConnection') as create,self.assertRaises(ValueError):
            peer.Session(deadline=time.monotonic()+5,ice_profile='turn:unapproved')
        create.assert_not_called()


if __name__=='__main__':unittest.main()
