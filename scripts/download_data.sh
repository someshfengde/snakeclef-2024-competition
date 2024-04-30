#!/bin/bash
download_dir="data"

# Create directory if it doesn't exist
mkdir -p "$download_dir"
# URLs to download
# all the URLS here are the small URLs
urls=(
  "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-train-small_size.tar.gz"
  "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-trainHMP.tar.gz"
  "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-val-small_size.tar.gz"
  "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-TrainMetadata-iNat.csv"
  "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-TrainMetadata-HM.csv"
  "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-ValMetadata.csv"
  "https://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/venomous_status_list.csv"
)

# Download each file using wget
for url in "${urls[@]}"; do
  wget --no-check-certificate -P "$download_dir" "$url"
done

cd "$download_dir"

# Find all .tar.gz files in the directory, extract them, and then delete the archive
echo "Extracting all .tar.gz files..."
for file in *.tar.gz; do
  echo "Extracting $file..."
  tar -xzvf "$file"
  echo "Deleting archive $file..."
  rm "$file"
done
echo "All files extracted and archives deleted."
