# -*- coding: utf-8 -*-
from python_codes.util.config import args
from python_codes.visualize.breast_cancer import *

if __name__ == "__main__":
    # export_data_pipeline(args,filtered=False)
    figure_pipeline(args)
    # basic_pipeline(args)
    # umap_pipeline(args)
    # go_pipeline(args)
    # expr_analysis_pipeline(args)
    # corr_expr_analysis_pipeline(args)
    # cell_cell_communication_prep_pipeline(args)