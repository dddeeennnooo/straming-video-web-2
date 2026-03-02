from src.data import make_dataset
from src.train import train
from src.predict import predict

df = make_dataset()

model = train(df)

p, odds = predict(df.tail(10))

print("\nPosledných 10 udalostí:")
for prob, o in zip(p, odds):
    print(f"p={prob:.3f}  odds={o:.2f}")
