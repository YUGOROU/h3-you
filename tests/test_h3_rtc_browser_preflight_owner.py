"""Owner/UI event contracts only; no browser, network, cloud or playback."""
import time
import unittest
from unittest.mock import Mock,patch
from research.run_h3_webrtc_connect import browser_probe,main

class BrowserOwnerTests(unittest.TestCase):
    def probe(self,events):
        proxy=Mock();proxy.url='http://127.0.0.1:1234/#private-capability'
        proxy.receipts=tuple(dict(event=e) for e in events)
        factory=Mock();factory.return_value.__enter__=Mock(return_value=proxy)
        factory.return_value.__exit__=Mock(return_value=False)
        return proxy,factory

    def test_private_callback_and_no_audibility_claim(self):
        proxy,factory=self.probe(('connected','frame-presented','unmuted','stop'))
        callback=Mock()
        with patch('research.h3_personal.webrtc_browser_proxy.BrowserProxy',factory):
            result=browser_probe(dict(expires_at=time.time()+30),'HF','APP',None,on_browser=callback)
        callback.assert_called_once_with(proxy.url)
        self.assertTrue(result['audio_enable_requested'])
        self.assertFalse(result['audibility_verified'])
        self.assertFalse(result['exact_media_verified'])
        self.assertNotIn('url',result)

    def test_stopping_early_and_failure_are_terminal(self):
        for events in (('stop',),('connected','error')):
            _,factory=self.probe(events)
            with patch('research.h3_personal.webrtc_browser_proxy.BrowserProxy',factory):
                with self.assertRaises(RuntimeError):
                    browser_probe(dict(expires_at=time.time()+30),'HF','APP',None,on_browser=Mock())

    def test_expired_wait_closes_proxy(self):
        _,factory=self.probe(())
        with patch('research.h3_personal.webrtc_browser_proxy.BrowserProxy',factory):
            with self.assertRaises(TimeoutError):
                browser_probe(dict(expires_at=time.time()-1),'HF','APP',None,on_browser=Mock())
        factory.return_value.__exit__.assert_called_once()

    def test_browser_offline_never_launches(self):
        with patch('research.run_h3_webrtc_connect.run',side_effect=AssertionError),patch('builtins.print'):
            self.assertEqual(main(['--browser','--hf-turn']),0)

if __name__=='__main__':unittest.main()
