from pathlib import Path

train_source = Path("data/Data/Train_Data")
test_source = Path("data/Data/Test_Data")

print("Train folder contents:")
for f in train_source.rglob("*.*"):
    print(f)

print("\nTest folder contents:")
for f in test_source.rglob("*.*"):
    print(f)
