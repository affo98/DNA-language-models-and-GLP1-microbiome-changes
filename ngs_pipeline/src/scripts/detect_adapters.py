import sys
import zipfile
import os

# Define adapter keywords from FastQC reports
ADAPTERS = {
    "TruSeq3": ["TruSeq3", "Illumina Universal Adapter"],
    "TruSeq2": ["TruSeq2"],
    "NexteraPE": ["Nextera"],
}


def detect_adapter(fastqc_zip):
    # Extract the FastQC output directory name
    fastqc_dir = fastqc_zip.replace(".zip", "")

    # Unzip if needed
    if not os.path.exists(fastqc_dir):
        with zipfile.ZipFile(fastqc_zip, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(fastqc_zip))

    # Read `fastqc_data.txt`
    fastqc_data = os.path.join(fastqc_dir, "fastqc_data.txt")
    if not os.path.exists(fastqc_data):
        # print(f"Error: {fastqc_data} not found")
        sys.exit(1)

    with open(fastqc_data, "r") as f:
        lines = f.readlines()

    # Look for Adapter Content section
    found_adapters = []
    for line in lines:
        for adapter_type, keywords in ADAPTERS.items():
            if any(keyword in line for keyword in keywords):
                # print(f"Found adapter{adapter_type} in {fastqc_data}")
                found_adapters.append(adapter_type)

    # Return the most specific adapter set
    if "NexteraPE" in found_adapters:
        return "NexteraPE"
    elif "TruSeq3" in found_adapters:
        return "TruSeq3"
    elif "TruSeq2" in found_adapters:
        return "TruSeq2"
    else:
        return "unknown"


if __name__ == "__main__":
    fastqc_zip = sys.argv[1]
    adapter = detect_adapter(fastqc_zip)
    print(adapter)
