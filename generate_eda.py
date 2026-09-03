import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exploratory Data Analysis: Edge-AUI Framework\n",
                "This notebook explores the preprocessed MicroTensor dataframes from all 4 foundational datasets to analyze kinematic features and validate the modality mask strategies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "import sys\n",
                "sys.path.append(\"../src\")\n",
                "from preprocessing import (\n",
                "    process_ck_session,\n",
                "    process_hvt_session,\n",
                "    process_hmi_sequences,\n",
                "    process_action_paths,\n",
                "    FEATURE_NAMES\n",
                ")\n",
                "\n",
                "DATA_ROOT = \"../.data/raw\""
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Continuous Kinematics 2020"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ck_logs = glob.glob(f\"{DATA_ROOT}/continuous-kinematics-2020/logs/*.csv\")[:3]\n",
                "ck_windows = []\n",
                "for log in ck_logs:\n",
                "    ck_windows.extend(process_ck_session(log))\n",
                "\n",
                "df_ck = pd.DataFrame([w[\"features\"] for w in ck_windows], columns=FEATURE_NAMES)\n",
                "df_ck[\"Dataset\"] = \"Continuous Kinematics\"\n",
                "df_ck.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. High-Volume Trajectories 20226"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "hvt_logs = glob.glob(f\"{DATA_ROOT}/high-volume-trajectories-20226/**/*.csv\", recursive=True)[:1]\n",
                "hvt_windows = []\n",
                "for log in hvt_logs:\n",
                "    hvt_windows.extend(process_hvt_session(log))\n",
                "\n",
                "df_hvt = pd.DataFrame([w[\"features\"] for w in hvt_windows], columns=FEATURE_NAMES)\n",
                "df_hvt[\"Dataset\"] = \"High-Volume Trajectories\"\n",
                "df_hvt.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Structural HMI Sequences 2023"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "hmi_logs = glob.glob(f\"{DATA_ROOT}/structural-hmi-sequences-2023/*.csv\")[:1]\n",
                "hmi_windows = []\n",
                "for log in hmi_logs:\n",
                "    hmi_windows.extend(process_hmi_sequences(log))\n",
                "\n",
                "df_hmi = pd.DataFrame([w[\"features\"] for w in hmi_windows], columns=FEATURE_NAMES)\n",
                "df_hmi[\"Dataset\"] = \"HMI Sequences\"\n",
                "df_hmi.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Client-Side Action Paths 2021"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ap_logs = glob.glob(f\"{DATA_ROOT}/client-side-action-paths-2021/**/*.json\", recursive=True)[:3]\n",
                "ap_windows = []\n",
                "for log in ap_logs:\n",
                "    ap_windows.extend(process_action_paths(log))\n",
                "\n",
                "df_ap = pd.DataFrame([w[\"features\"] for w in ap_windows], columns=FEATURE_NAMES)\n",
                "df_ap[\"Dataset\"] = \"Action Paths\"\n",
                "df_ap.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Feature Distributions (Velocity & Hesitation)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Combine datasets for distribution analysis\n",
                "df_all = pd.concat([df_ck, df_hvt, df_hmi, df_ap], ignore_index=True)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "sns.kdeplot(data=df_all, x=\"meanVelocity\", hue=\"Dataset\", fill=True, ax=axes[0], common_norm=False)\n",
                "axes[0].set_title(\"Distribution of Mean Velocity\")\n",
                "\n",
                "sns.kdeplot(data=df_all, x=\"hesitationCount\", hue=\"Dataset\", fill=True, ax=axes[1], common_norm=False)\n",
                "axes[1].set_title(\"Distribution of Hesitation Count\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Cross-Dataset Feature Discrepancy Matrix (Correlation Heatmap)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "corr_matrix = df_all.drop(columns=[\"Dataset\"]).corr()\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(corr_matrix, annot=True, cmap=\"coolwarm\", fmt=\".2f\")\n",
                "plt.title(\"Cross-Dataset Feature Correlation Matrix\")\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/EDA.ipynb", "w") as f:
    json.dump(notebook, f, indent=4)
