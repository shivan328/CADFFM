import numpy as np
import sys
sys.path.append("./src")
import indicators as ind  # NOQA
import util as ut  # NOQA

cadffe_wd_file = './results_indicator/bi_420_ic_0.005_hf_0.09_max_all.tif'
reference_wd_file = './results_indicator/Max_Depth_Maz_Corrected.tif'
cadffm_wd_file = './results_indicator/Scotchmans_Creek_out_mwd_CADFFM.tif'

predicted_data, mask, bounds, cell_size = ut.RasterToArray(cadffm_wd_file)
observed_data, mask, bounds, cell_size = ut.RasterToArray(reference_wd_file)

pm = ind.PerformanceIndicators(predicted_data, observed_data, 0.005)
pm.print_indicators()
