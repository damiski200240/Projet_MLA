import torch
from Decoder import Decoder  

def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)

   
    B = 4          
    n_attr = 3     

    
    z = torch.randn(B, 512, 2, 2, device=device)         
    y = torch.randint(0, 2, (B, n_attr), device=device).float()  

    
    decoder = Decoder(n_attr=n_attr).to(device)

    
    x_hat = decoder(z, y)

    
    print("z shape     :", z.shape)       
    print("y shape     :", y.shape)       
    print("x_hat shape :", x_hat.shape)   
if __name__ == "__main__":
    main()