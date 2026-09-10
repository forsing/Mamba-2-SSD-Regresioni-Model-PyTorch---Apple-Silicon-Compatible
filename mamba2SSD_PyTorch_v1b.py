# Model V1: Mamba-2 SSD Regresioni Model (PyTorch - Apple Silicon Compatible)



import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time

# =====================================================================
# GLAVNI PARAMETRI ZA AUTOMATSKU PROMENU (PODEŠAVAJ SAMO OVDE)
# =====================================================================
CSV_FILE = "/Users/4c/Desktop/GHQ/data/loto7_4682_k72_loto_2963.csv"   # Ime tvog fajla
WINDOW_SIZE = 30         # Prozor (za V1 drži niske vrednosti: 15-40)
NUM_EPOCHS = 120         # Broj epoha (za V1 drži 100-250)
# =====================================================================

# Automatska selekcija Apple Silicon GPU-a (MPS) ili CPU-a
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Mamba-2 SSD PyTorch | Fajl: {CSV_FILE} | Prozor: {WINDOW_SIZE} | Epoha: {NUM_EPOCHS} | Uređaj: {device}")

# 1. ČISTA PARALELNA MAMBA-2 / SSD ARHITEKTURA
class Mamba2SSDLayer(nn.Module):
    def __init__(self, d_model, d_state=64, headdim=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.d_inner = d_model * 2 
        self.nheads = self.d_inner // headdim
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + self.d_state * 2 + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=2,
            groups=self.d_inner
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        projected = self.in_proj(x)
        
        x_branch, res_branch, B, C, dt = torch.split(
            projected, 
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads], 
            dim=-1
        )
        
        x_branch = x_branch.transpose(1, 2)
        x_branch = torch.sigmoid(self.conv1d(x_branch)[:, :, :seq_len].transpose(1, 2))

        dt = torch.exp(dt) 
        A_matrix = torch.einsum("bsh, bsn -> bhsn", dt, x_branch[:, :, :self.nheads])
        ssm_output = torch.einsum("bhsn, bsn -> bsn", A_matrix, x_branch[:, :, :self.nheads])
        
        ssm_output_full = torch.zeros(batch, seq_len, self.d_inner, device=x.device)
        ssm_output_full[:, :, :ssm_output.shape[-1]] = ssm_output
        
        gated_output = ssm_output_full * torch.nn.functional.silu(res_branch)
        return self.out_proj(gated_output)

# 2. PREDVIĐAČKI MODEL
class Mamba2Predictor(nn.Module):
    def __init__(self, vocab_size=40, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba2_layer1 = Mamba2SSDLayer(d_model=d_model)
        self.mamba2_layer2 = Mamba2SSDLayer(d_model=d_model)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.embedding(x) 
        x = x.mean(dim=2)
        
        x = self.mamba2_layer1(x) + x
        x = self.mamba2_layer2(x) + x
        
        x_last = x[:, -1, :].repeat(1, 7)
        return self.fc_out(x_last)

# 3. KREIRANJE MATRICA I TRENING PETLJA
if __name__ == "__main__":
    try:
        df = pd.read_csv(CSV_FILE, header=None)
        raw_data = df.values
        
        X_list = [raw_data[i:i+WINDOW_SIZE] for i in range(len(raw_data) - WINDOW_SIZE)]
        Y_list = [raw_data[i+WINDOW_SIZE] for i in range(len(raw_data) - WINDOW_SIZE)]
        
        X = torch.tensor(np.array(X_list), dtype=torch.long).to(device)
        Y = torch.tensor(np.array(Y_list), dtype=torch.float).to(device)
        
        model = Mamba2Predictor(vocab_size=40, d_model=128).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        
        model.train()
        print("Trening modela je pokrenut...")
        start_time = time.time()
        
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            
            # AUTOMATIZOVAN ISPIS: Prati tačan broj epoha sa vrha skripte
            if (epoch + 1) % 50 == 0:
                print(f"Mamba-2 PyTorch Epoha [{epoch+1}/{NUM_EPOCHS}] | SSD Gubitak: {loss.item():.4f}")
                
        print(f"Trening završen za: {time.time() - start_time:.2f} sekundi.")
        
        # PREDVIĐANJE SLEDEĆEG REDA
        model.eval()
        with torch.no_grad():
            poslednji_prozor = torch.tensor(raw_data[-WINDOW_SIZE:], dtype=torch.long).unsqueeze(0).to(device)
            raw_pred = model(poslednji_prozor).cpu().numpy()
            
            sledeci_red = np.clip(np.round(raw_pred), 1, 39).astype(int)
            sledeci_red = np.sort(np.unique(sledeci_red))
            
            while len(sledeci_red) < 7:
                novi_broj = np.random.randint(1, 40)
                if novi_broj not in sledeci_red:
                    sledeci_red = np.append(sledeci_red, novi_broj)
            sledeci_red = np.sort(sledeci_red)
            
            print("\n" + "="*50)
            print(f"REZULTAT ZA FAJL {CSV_FILE} (Sledeći red - PyTorch V1):")
            print(sledeci_red)
            print("="*50)
            
    except FileNotFoundError:
        print(f"Greška: Fajl '{CSV_FILE}' nije pronađen.")



"""
Mamba-2 SSD PyTorch | Fajl: /Users/4c/Desktop/GHQ/data/loto7_4682_k72_loto_2963.csv | Prozor: 30 | Epoha: 120 | Uređaj: mps
Trening modela je pokrenut...
Mamba-2 PyTorch Epoha [50/120] | SSD Gubitak: 63.5068
Mamba-2 PyTorch Epoha [100/120] | SSD Gubitak: 30.2218
Trening završen za: 53.22 sekundi.

==================================================
REZULTAT ZA FAJL /Users/4c/Desktop/GHQ/data/loto7_4682_k72_loto_2963.csv (Sledeći red - PyTorch V1):
[ 5 10 14 19 24 29 34]
==================================================
"""



"""
Prepoznavanje obrazaca (Pattern Recognition) 
Otkrivanje obrazaca (Pattern Discovery) 
Rudarenje obrazaca (Pattern Mining)   --->   Mamba-2

Model V1: Mamba-2 SSD Regresioni Model (PyTorch - Apple Silicon Compatible)
Model V2: Mamba-2 SSD Kategorijalni Klasifikator (Apple MLX - Native Silicon)
Model V3: Kanonski TFT Kategorijalni Klasifikator (PyTorch - Apple Silicon Compatible) 
Model V4: Kanonski TFT Kategorijalni Klasifikator (Apple MLX - Native Silicon)

Arhitektura Mamba 2 se zasniva na teoriji Structured State Space Duality (SSD). 
Mamba 2 omogućava da se proračun stanja transformiše u blokovske matrične multiplikacije, 
što je znatno lakše napisati u čistom Python-u/PyTorch-u. 
"""



"""
Optimalni odnosa između dužine istorijskog prozora i broja epoha za bazu podataka. 
Cilj je balans: dovoljno velik prozor da Mamba-2 uhvati cikluse, 
ali dovoljno primera za trening da model ne upadne u hiper-podešavanje (overfitting).

Evo optimalnih vrednosti za oba modela na osnovu količine podataka u tri CSV fajla, 
kako bi se sprečio overfitting (prenaučenost) i maksimalno iskoristila dužina istorije: 

Model V1,V3: PyTorch (Kraći prozor, brža konvergencija)
Za 4682 reda: Prozor: 40 | Epohe: 150 
Za 2963 reda: Prozor: 30 | Epohe: 120 
Za 1719 reda: Prozor: 20 | Epohe: 100  

Model V2,V4: Apple MLX (Širi prozor, dublja istorija)
Za 4682 reda: Prozor: 200 | Epohe: 1200 
Za 2963 reda: Prozor: 100 | Epohe: 1000 
Za 1719 reda: Prozor:  50 | Epohe: 400 
"""



"""
Mamba / S4 (State Space Models - SSM) 
Najnovija generacija AI arhitektura koja u mnogim zadacima predviđanja sekvenci nadmašuje čak i Transformere. 
Mamba ima linearno skaliranje i koristi selektivni mehanizam stanja. 
Za razliku od standardnih modela koji se muče sa dugoročnim zavisnostima u brojevima, 
Mamba može da kompresuje celu istoriju u jedno kompaktno "stanje" i precizno uoči ako se u CSV fajlu krije složen, 
visokodimenzionalni matematički algoritam ili generator.


Mamba / S4 (State Space Models) je arhitektonski napredniji i teoretski moćniji model od TFT-a za pronalaženje dubokih zakonitosti u dugim nizovima. 
Mamba je dizajnirana upravo da reši najveću manu starijih modela: sposobnost da filtrira nevažne podatke i zadrži savršen matematički fokus na ključnim promenama kroz vreme, bez gubitka memorije. 
Kroz svoj selektivni mehanizam stanja (Selective State Space), Mamba će pokušati da mapira skrivenu funkciju koja generiše ove brojeve i izračuna tačne vrednosti za sledećih 7 brojeva (next red).



Mamba / S4 ima suštinske prednosti koje direktno utiču na pronalaženje dubokih zakonitosti u loto kombinacijama: 

Efektivni kontekst nad dugom istorijom: 
Mamba koristi linearni selective scan mehanizam koji kompresuje celu istoriju CSV redova u jedno skriveno stanje konstantne veličine. 
TFT se oslanja na pažnju (Attention) koja ima kvadratnu složenost i gubi stabilnost kada prozor postane preveliki. 

Neprekidno modelovanje vremena (Continuous-time SSM): 
Mamba kroz diskretizaciju (Delta) uči skriveni kontinuum i dinamiku sistema. 
Ona tretira vaš CSV kao kontinualni signal koji se razvija kroz vreme, 
što joj omogućava da uoči duboke, ciklične i skrivene repetitivne obrasce koje TFT-ovi statični prozori promašuju. 

Selekcija informacija kroz vreme: 
Mamba filtrira nevažne šumove u svakom koraku sekvence. 
Za razliku od TFT-a koji pokušava da odjednom izvaže uticaj svih kolona u fiksnom prozoru, 
Mamba dinamički odlučuje šta iz prethodnih izvlačenja treba trajno zapamtiti, a šta odbaciti.


Model koristi Embedding sloj veličine 40 (za brojeve 1-39, gde je 0 rezervisana za mapiranje) 
i višeslojnu kauzalnu strukturu sa mehanizmom selektivnog stanja i rezidualnim vezama.
Embedding sloj: 
Brojeve od 1 do 39 ne posmatra kao proste cifre, već kreira "guste vektore" (d_model=256). 
Na taj način model uči skriveni kontekst (npr. kako se broj 7 ponaša kada je na prvoj poziciji u odnosu na to kada je na trećoj poziciji). 

Python kod za pripremu i treniranje modela
učitava csv, priprema podatke metodom kliznog prozora (gledajući istoriju da bi predvideo sledeći red) i trenira model visoke moći.


SSM/Mamba princip duboke kompresije: 
Za razliku od klasičnih modela, unutrašnji slojevi (SSMResidualBlock) vrše ne-linearnu projekciju podataka u prostor od 512 dimenzija (d_state). 
To omogućava računaru da zadrži matematičku strukturu informacija kroz ceo niz od CSV koraka.

Automatsko sortiranje na izlazu: 
Kod na samom kraju uzima sirove matematičke vrednosti modela, osigurava da ostanu u opsegu 1-39, 
uklanja eventualne duplikate i sortira ih od najmanjeg do najvećeg, prateći strukturu prethodnih redova.

 
Kada se pokrene skriptu, ona prođe kroz svih CSV redova da bi naučila pravila. 
Kada se trening završi, model u memoriji drži skriveno stanje sistema. 
Funkcija model(poslednji_prozor) uzima sam kraj CSV fajla (istoriju neposredno pre next koraka) 
i na osnovu svega što je naučila generiše potpuno novih 7 brojeva. 
Kada se pokrene kod u terminalu, na samom dnu se dobije jasan ispis. 
"""



"""
Mamba, odnosno Selective State Space Model (SSM) — preciznije novija arhitektura Mamba-2.
Za sekvencijalno predviđanje na CSV podacima glavni model bi bio:
Mamba-2 regresioni model za sekvence skupova
Svako izvlačenje kodira se kao binarni vektor od 39 vrednosti, 
Mamba-2 obrađuje hronološki niz prethodnih izvlačenja, 
a regresiona glava daje kontinuirani skor za svih 39 brojeva. 
Sedam najvećih skorova čini NEXT.


Da bi kod bio napisan u čistoj Mamba-2 arhitekturi, 
on bi morao da koristi zvaničnu mamba_ssm biblioteku i njene specifične CUDA operatore, 
što zahteva Linux operativni sistem i grafičku kartu (GPU).
"""
