import unittest
from types import SimpleNamespace as NS
from research.h3_personal.fast_ref2va import FastRef2VAConfig,config_for_layout
from research.h3_personal.ref2va_runtime_options import export_geometry,finish_visual_only,residency_receipt


class RuntimeOptionsTests(unittest.TestCase):
    def test_portrait_canvas_and_export_are_distinguished(self):
        g=export_geometry('portrait');c=config_for_layout('portrait')
        self.assertEqual((c.width,c.height),(256,448))
        self.assertEqual(g['width']*16,g['height']*9)
        self.assertEqual(g['canvas_width']-g['crop_left']-g['crop_right'],g['width'])
        self.assertEqual(c.width*c.height,FastRef2VAConfig().width*FastRef2VAConfig().height)
        with self.assertRaises(ValueError):FastRef2VAConfig(width=252,height=448).validate()
        with self.assertRaises(ValueError):export_geometry('arbitrary')

    def test_visual_finish_preserves_video_but_cleans_native_runtime(self):
        image=object();batch=NS(output=image,extra={},audio_latents=object())
        def clear(batch):batch.audio_latents=None
        stage=NS(audio_vae=NS(sampling_rate=24000),_clear_runtime=clear)
        torch=NS(float32='float32',empty=lambda shape,**kw:('empty',shape))
        self.assertIs(finish_visual_only(batch,stage,torch).output,image)
        self.assertIsNone(batch.audio_latents)
        self.assertEqual(batch.extra['audio'],('empty',(0,2)))
        self.assertEqual(batch.extra['audio_sample_rate'],24000)

    def test_residency_receipt_rejects_cpu_weights(self):
        module=NS(parameters=lambda:iter([NS(device='cpu')]),buffers=lambda:iter([]))
        pipeline=NS(get_module=lambda key:module)
        with self.assertRaisesRegex(RuntimeError,'resident'):residency_receipt(pipeline,NS())

    def test_prepared_reference_cache_binds_bytes_and_copies_metadata(self):
        import tempfile
        from pathlib import Path
        from research.h3_personal.ref2va_runtime_options import prepare_images_cached
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'reference';path.write_bytes(b'photo')
            calls=[];pixels=object();stage=NS();refs=[NS(media_type='image',source=path)]
            def prepare(r,n,s):calls.append((n,s));return NS(image=pixels,num_latent_frames=None)
            first=prepare_images_cached(stage,refs,124,24000,prepare);first[0].num_latent_frames=1
            second=prepare_images_cached(stage,refs,124,24000,prepare)
            self.assertEqual(len(calls),1);self.assertIs(second[0].image,pixels)
            self.assertIsNone(second[0].num_latent_frames)
            self.assertEqual(stage._smd_prepared_image_hits,1)
            path.write_bytes(b'new photo');prepare_images_cached(stage,refs,124,24000,prepare)
            self.assertEqual(len(calls),2)
            prepare_images_cached(stage,refs,141,24000,prepare);self.assertEqual(len(calls),3)

    def test_resident_generator_policy_is_opt_in(self):
        from research.h3_personal.fast_ref2va import generator_config
        api=NS(**{key:NS for key in ('GeneratorConfig','PipelineSelection','ComponentConfig','EngineConfig','ParallelismConfig','OffloadConfig','CompileConfig')})
        self.assertTrue(generator_config(api,'base','adapter').engine.offload.vae)
        candidate=generator_config(api,'base','adapter',runtime_policy='resident-visual-cached')
        self.assertFalse(candidate.engine.offload.vae);self.assertFalse(candidate.engine.compile.enabled)

    def test_native_clear_runtime_not_reimplemented(self):
        from pathlib import Path
        source=Path('research/h3_personal/fast_ref2va.py').read_text()
        self.assertIn("name=='audio_decoding_stage'",source)
        self.assertIn('finish_visual_only(batch,pipeline._stage_name_mapping[name],torch)',source)

if __name__=='__main__':unittest.main()
