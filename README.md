# VoronoiAutoEncoder

The goal of this project is to build a network that is able to pick colors for given input pixel coordiantes that best reconstruct an image given a visualization of the corresponding [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram). So essentially it is as odd (and not very useful) autoencoder that instead of a latent space uses a set of given pixel coordinates
and a corresponding list of colors. The decoder is then just the construction of the diagram. 



## Implementation
At training the set of coordinates are generated at random and the output is scored using the L1 distance from the original image. One problem that became clear whilst working on this is that since the model is required to match colors to given coordinates it needs to have accurate positional information. My first attempts using a ResNet failed for variable points and worked for static points and this may have been the reason. 

In order to account for this a small network along the lines of a [vision transformer](https://arxiv.org/abs/2010.11929) is used. The image is split into patches, each patch is flattened, a learned positional encoding is added and the result is mapped to a set embedding space. From each of the patch embeddings a key, value pair is produced.
Each input coordinate is also mapped to the same embedding dimension and a query is generated for each. Finally cross attention is computed resulting in a final embedding for each coordinate which is then mapped to a color using a MLP.

The model is trained with a low batch size of 4 for 12 epochs (about 2h) on the celeba hq 256 dataset but would likely work as good or better (less detailed images likely a better fit) on other datasets.


![Result example.]("https://github.com/SamGalanakis/VoronoiAutoEncoder/blob/bead518a7374ea8da79e3e2e3bc0e5139d9b1784/result.png")