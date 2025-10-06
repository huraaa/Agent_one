CASES = [
  # RAG
  {"q":"What is section 3.1 about? Cite the source path.", "expect_src":"docs/", "judge_grounded": True},
  {"q":"Summarize section 2.1 in one sentence with the file path.", "expect_src":".pdf", "judge_grounded": True},

  # Math/tool
  {"q":"Compute (17*24)+5 and return only the number.", "expect_ans":"413"},

  # Web (requires TAVILY_API_KEY or your requests fallback)
  {"q":"List two AI code search tools with URLs.", "expect_src":"http", "judge_grounded": False},

  # JSON format check
  {"q":"Return a JSON object with keys name and city: name=Alice, city=Toronto.", "require_json": True,
   "expect_ans": "{\"name\": \"Alice\", \"city\": \"Toronto\"}"},

  # Mixed tools
  {"q":"From our PDFs, give one-sentence summary of section 3.1 with path, then compute 250*1.13." , "expect_src":"docs/"},

  # Memory usage (after you save profile)
  {"q":"What citation style do I prefer?", "expect_ans":"path-only"},

  # Sentiment tool (optional if added)
  {"q":"Classify sentiment: \"This movie was fantastic and moving.\"", "expect_ans":"positive"},

  # Another RAG
  {"q":"Quote a key term from section 2.1 and cite the PDF path.", "expect_src":"docs/", "judge_grounded": True},

  # Another JSON validity
  {"q":"Output JSON with keys a and b where a=3 and b=7.", "require_json": True,
   "expect_ans": "{\"a\": 3, \"b\": 7}"},
]
