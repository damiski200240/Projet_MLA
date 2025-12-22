import torch
from Encoder import Encoder   

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)

    encoder = Encoder().to(device)

    x = torch.randn(4, 3, 256, 256, device=device)

    z = encoder(x)

    print("Shape entrée  x :", x.shape)  
    print("Shape sortie  z :", z.shape)  

if __name__ == "__main__":
    main()