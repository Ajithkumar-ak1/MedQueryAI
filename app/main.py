from generator import ask_medivault

while True:

    query = input("\nAsk Medical Question: ")

    if query.lower() in ["exit", "quit"]:
        break

    result = ask_medivault(query)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\nSOURCES")

    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"(Page {source['page']})"
        )