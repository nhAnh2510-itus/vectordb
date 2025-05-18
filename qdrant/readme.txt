dowload drant image: docker pull qdrant/qdrant

init: docker run -d --name qdrant-container -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant


sau khi tat muon mo lai: docker start qdrant-container
