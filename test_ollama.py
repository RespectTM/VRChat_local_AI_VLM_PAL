from ollama_client import run_ollama

def main():
    prompt = "You are a helpful assistant. Say hi briefly."
    print("Sending prompt to model: ", prompt)
    out = run_ollama("vrchat-moondream2", prompt)
    print("--- Model output ---")
    print(out)
    print("--------------------")

if __name__ == '__main__':
    main()
