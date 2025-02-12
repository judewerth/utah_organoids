# Standard Operating Procedure for Utah Organoids DataJoint pipeline

## Overview

This document provides a step-by-step guide for accessing and using the **Utah Organoids DataJoint pipeline**. The pipeline is designed to manage and analyze data from the Utah lab, focusing on cerebral organoids characterization and electrophysiology (ephys) data analysis.

- **Organoid Generation Pipeline**: This pipeline manages metadata for organoid generation protocols, including inducing pluripotent stem cells (iPSCs) to form single neural rosettes (SNRs), which then develop into organoids.

- **Array Ephys Pipeline**: This pipeline handles the analysis of array ephys data, managing metadata and raw data related to probes and ephys recordings. It also includes preprocessing, spike sorting, curation, and quality metrics computations.

## Accessing the Utah Organoids DataJoint Pipeline

1. Request access and account at [DataJoint Works account](https://accounts.datajoint.com/).
     a. Contact DataJoint team for access & account creation.
     b. Once approved, you will receive **DataJoint credentials (username and password)** to access the DataJoint Works account, the Organoids SciViz website, and to connect to the database locally.

## Standard Operating Procedure for the Organoids Generation Pipeline

2. Enter metadata into the **Organoids Generation Pipeline**. Please manually input relevant data using the provided entry forms on the website as follows:
     a. Visit the [Organoids SciViz website](https://organoids.datajoint.com/) and log in with your DataJoint credentials (username and password)
     b. Follow the data-entry steps in the "Form" sections of each tab to specify the details of your organoid generation protocol:
        i. `User` page → if you are a new experimenter, create a new user
        ii. `Linage` page → create new “Linage” and submit; create new “Sequence” and submit
        iii. `Stem Cell` page → create new “Stem Cell”
        iv. `Induction` page → add new “Induction Culture” and “Induction Culture Condition”
        v. `Post Induction` page → add new “Post Induction Culture” and “Post Induction Culture Condition”
        vi. `Isolated Rossette` page → add new “Isolated Rossette Culture” and “Isolated Rossette Culture Condition”
        vii. `Organoid` page → add new “Organoid Culture” and “Organoid Culture Condition”
        viii. `Experiment` page → add new experiments performed on a particular organoid
            1. Include organoids ID, datetime, experimenter, condition, etc.
            2. Provide the experiment data directory — the relative path to where the acquired data is stored.
Note: The "Table" sections in each tab display the data entries in a tabular format. These sections are not clickable. If you click on them, the website may turn white, requiring you to log back in.

## Standard Operating Procedure for the Array Ephys Pipeline

3. Select an organoid experiment and define a time-window for ephys analysis (referred to as `EphysSession` in the pipeline)
    a. Go to `works.datajoint.com` → `Notebook` tab
    b. Open [this notebook to create a new `EphysSession`](https://github.com/dj-sciops/utah_organoids/blob/main/notebooks/CREATE_new_session.ipynb) and follow the instructions.
        i. For LFP analysis, define the `session_type` as `lfp` (details are provided in the notebook). The `EphysSession` for LFP analysis will trigger the analysis automatically after creation.
        ii. For spike sorting analysis, define the `session_type` as `spike_sorting`, and also create a `EphysSessionProbe` to store probe information, including the channel mapping (details are provided in the notebook). The `EphysSession` and `EphysSessionProbe` will trigger probe insertion detection automatically. For spike sorting, you will need to manually select the spike sorting algorithm and parameter set to run (see the next step).
4. Ephys spike sorting analysis
    a. The user must manually select which spike-sorting algorithm and parameter set to run (this is called to create a `ClusteringTask` in the pipeline)
        i. Go to `works.datajoint.com` → `Notebook` tab
        ii. Open [this notebook to create a new spike sorting parameter set (clustering paramset) for an `EphysSession`](https://github.com/dj-sciops/utah_organoids/blob/main/notebooks/CREATE_new_clustering_paramset.ipynb) and follow the instructions.
        iii. Open [this notebook to select the spike sorting parameter set and the `EphysSession`](https://github.com/dj-sciops/utah_organoids/blob/main/notebooks/CREATE_new_clustering_task.ipynb)(i.e., a `ClusteringTask` in the pipeline) and follow the instructions.
        iv. Spike sorting will trigger automatically after your selection.
5. Explore LFP/ spike sorting results 
    a. Go to `works.datajoint.com` → `Notebook` tab
        i. Open and follow the instructions in [this notebook here to explore the ephys results in the pipeline](https://github.com/dj-sciops/utah_organoids/blob/main/notebooks/EXPLORE_array_ephys.ipynb)
        ii. Then, open and follow the instructions in [this notebook here to explore the quality metrics for sorted units](https://github.com/dj-sciops/utah_organoids/blob/main/notebooks/EXPLORE_quality_metrics.ipynb)

### Upload Data from your Local Machine to the Cloud
1. Ensure that the local folder you want to upload follows the [file structure guidelines here](https://github.com/dj-sciops/utah_organoids/blob/main/docs/DATA_ORGANIZATION.md).
2. You need Axon credentials to upload raw data from you local machine to the cloud. If you don't have them yet:
    a. Request them from the DataJoint team.
    b. Once approved, you’ll be provided with **Axon credentials** (account_id, client_id, client_secret, issuer, bucket, role) to upload to the AWS S3 bucket.
3. Set up the following configurations on your local machine (if you haven't already):  
    a. Install the pipeline code on your computer. Follow the [installation instructions here](https://github.com/dj-sciops/utah_organoids/blob/main/docs/INSTALLATION_AND_CONFIGURATION_INSTRUCTIONS.md#installation-of-the-pipeline-codebase).  
    b. Set up the axon configuration by following the steps in this guide: [Add cloud upload configuration](https://github.com/dj-sciops/utah_organoids/blob/main/docs/CLOUD_UPLOAD_CONFIGURATION_INSTRUCTIONS.md).  
4. Once the setup is complete, you can upload your local raw data by following the steps in [this notebook](https://github.com/dj-sciops/utah_organoids/blob/main/notebooks/CREATE_new_session_with_cloud_upload.ipynb)

### Download Data from the Cloud to your Local Machine
1. You need Axon credentials to download the spike sorting results from the cloud to your local machine. If you don't have them yet:
    a. Request them from the DataJoint team.
    b. Once approved, you’ll be provided with **Axon credentials** (account_id, client_id, client_secret, issuer, bucket, role) to download results from the AWS S3 bucket.
2. Set up the following configurations on your local machine (if you haven't already):  
    a. Install the pipeline code on your computer. Follow the [installation instructions here](https://github.com/dj-sciops/utah_organoids/blob/main/docs/INSTALLATION_AND_CONFIGURATION_INSTRUCTIONS.md#installation-of-the-pipeline-codebase).  
    b. Set up the axon configuration by following the steps in this guide: [Add cloud upload configuration](https://github.com/dj-sciops/utah_organoids/blob/main/docs/CLOUD_UPLOAD_CONFIGURATION_INSTRUCTIONS.md). 
3. Once the setup is complete, you can download the spike sorting results by following the steps in [this notebook](https://github.com/dj-sciops/utah_organoids/blob/main/notebooks/DOWNLOAD_spike_sorted_data.ipynb).