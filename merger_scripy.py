"""
Merges clinical and metadata embed csv files while taking into account
the side of the findings.
"""


if __name__ == "__main__":

    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm

    EMBED_ROOT = Path("/vol/biomedic3/data/EMBED")

    dicom = pd.read_csv(
        EMBED_ROOT / "tables/EMBED_OpenData_metadata.csv", low_memory=False
    )

    embed_clinical = pd.read_csv(
        EMBED_ROOT / "tables/EMBED_OpenData_clinical.csv", low_memory=False
    )

    result = embed_clinical.copy()
    both_or_nan = embed_clinical.loc[embed_clinical.side.isna() | (embed_clinical.side == "B")]
    for id, row in tqdm(both_or_nan.iterrows()):
        rl = row.copy()
        rl.side = "L"
        rr = row.copy()
        rr.side = "R"
        result = pd.concat([result, rl.to_frame().transpose(), rr.to_frame().transpose()], axis=0, ignore_index=True)

    # Rename column in meta to match clinical
    dicom.rename(columns={"ImageLateralityFinal": "side"}, inplace=True)

    merged = pd.merge(dicom, result, on=["acc_anon", "side", "empi_anon"], how="left")

    # Add image path
    merged["image_path"] = (
        merged["empi_anon"].astype("str")
        + "/"
        + merged["anon_dicom_path"].str.split("/").str[-1].str.split(".dcm").str[0]
        + ".png"
    )

    merged.to_csv("merged_df.csv")