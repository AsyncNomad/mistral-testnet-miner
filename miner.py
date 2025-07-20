from mistral_runner import generate_response

def main():
    print("Mistral Prompt Test ")
    print("Input('exit' to quit): \n")

    while True:
        prompt = input(">>> ")
        if prompt.lower() in ["exit", "quit"]:
            print("Test Ended")
            break
        response = generate_response(prompt)
        print(f"\n{response}\n")

if __name__ == "__main__":
    main()
