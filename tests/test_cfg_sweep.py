import unittest

from JiT.eval import cfg_sweep as cfg_sweep_module


class CfgSweepHelperTests(unittest.TestCase):
    def test_build_initial_cfg_grid_clamps_and_dedupes(self):
        grid = cfg_sweep_module.build_initial_cfg_grid(
            base_cfg=2.9,
            cfg_min=2.5,
            cfg_max=3.2,
            coarse_offsets=[-0.6, -0.4, 0.0, 0.4, 0.6],
        )

        self.assertEqual(grid, [2.5, 2.9, 3.2])

    def test_build_refine_cfg_grid_skips_existing_values(self):
        refined = cfg_sweep_module.build_refine_cfg_grid(
            best_cfg=2.9,
            evaluated_cfgs={2.4, 2.65, 2.9, 3.15, 3.4},
            cfg_min=1.5,
            cfg_max=4.0,
            refine_step=0.1,
            refine_count=2,
        )

        self.assertEqual(refined, [2.7, 2.8, 3.0, 3.1])

    def test_choose_best_result_prefers_fid_then_is_then_center_distance(self):
        results = [
            {"cfg": 2.8, "fid": 4.2, "inception_score": 180.0},
            {"cfg": 2.9, "fid": 4.0, "inception_score": 175.0},
            {"cfg": 3.0, "fid": 4.0, "inception_score": 181.0},
            {"cfg": 3.1, "fid": 4.0, "inception_score": 181.0},
        ]

        best = cfg_sweep_module.choose_best_result(results, center_cfg=3.05)

        self.assertEqual(best["cfg"], 3.0)


if __name__ == "__main__":
    unittest.main()
