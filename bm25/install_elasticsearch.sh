mkdir -p external_tools
cd external_tools

# download ElasticSearch
version=6.5.3
echo "Downloading ElasticSearch"
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${version}.tar.gz
tar -xf elasticsearch-${version}.tar.gz
rm elasticsearch-${version}.tar.gz
cd elasticsearch-${version}
echo "Installing additional plugins"
./bin/elasticsearch-plugin install analysis-icu
./bin/elasticsearch-plugin install analysis-kuromoji
./bin/elasticsearch-plugin install analysis-nori
./bin/elasticsearch-plugin install analysis-smartcn
./bin/elasticsearch-plugin install analysis-ukrainian
./bin/elasticsearch-plugin install analysis-stempel