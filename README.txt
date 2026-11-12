GMPP quarterly download pipeline (example files)

The full setup guide is GMPP_Quarterly_Download_Pipeline_Guide.docx
in this folder, and also at /workspace/gmpp/GMPP_Quarterly_Download_Pipeline_Guide.docx.

These scripts are meant to be installed at ~/GMPP-pipeline on the Mac.
They do not deploy a cloud service on their own.

Quick start on the Mac, after install:

  venv/bin/python3 gmpp_pipeline.py --print-paths
  venv/bin/python3 gmpp_pipeline.py --dry-run

A dry run never uploads a model and never writes into Desktop/gmpp-data.
