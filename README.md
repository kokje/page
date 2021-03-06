## PAGe - Privacy Aware Genomics
This framework allows users to experiment with privacy preserving federated learning for genomic datasets. Organizations or research groups can collaborate on learning a model for disease risk prediction without sharing user data with each other.  

## Usage
Ensure your working directory is set to the page parent folder and your raw vcf live under data/ 

### Dependencies
Docker 

### Build
You can build as many clients as you would like using:  
`docker build -t client1 --build-arg datafile=data/silo1.vcf.bgz .`  
`docker build -t client2 --build-arg datafile=data/silo2.vcf.bgz .`

### Run
PAGe uses the flask web framework so running the container will launch the program with default host set to localhost and port 5000. This can be configured either by modifying the code or updating commandline options in the dockerfile.   

When testing on your laptop you can map the containers default port to your device and test multiple clients. For ex:  
`docker run -it -d --name c1 -p 5001:5000 client1`  
`docker run -it -d --name c2 -p 5002:5000 client2`

Once the clients are running, you can launch an instance of your server by simply running:
`python server.py`

Please update the server configs to include the IP address and port for each of the clients to enable federation.






