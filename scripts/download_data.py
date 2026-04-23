# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

from pathlib import Path
import urllib.request


CSV_URL = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
PDF_URL = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "Ghana_Election_Result.csv"
    pdf_path = data_dir / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"

    print("Downloading CSV dataset...")
    urllib.request.urlretrieve(CSV_URL, csv_path)
    print(f"Saved: {csv_path}")

    print("Downloading PDF dataset...")
    urllib.request.urlretrieve(PDF_URL, pdf_path)
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
