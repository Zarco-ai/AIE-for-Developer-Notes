"""
This file is here so I can place code that I may or may not use!
"""
s = "soap"
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=s
)
embedding = response.data[0].embedding

# Print summary info
print(f"Total dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Fitness & Outdoors
    {"name": "Yoga Mat", "description": "Extra thick non-slip eco-friendly mat for yoga and pilates.", "category": "Fitness"},
    {"name": "Adjustable Dumbbells", "description": "Space-saving strength training equipment for home workouts.", "category": "Fitness"},
    {"name": "Hydration Backpack", "description": "Lightweight 2L water bladder for long distance running and hiking.", "category": "Fitness"},
    
    # Home & Kitchen
    {"name": "Air Fryer", "description": "Rapid air circulation technology for healthy oil-free cooking.", "category": "Kitchen"},
    {"name": "Espresso Machine", "description": "15-bar pump pressure system for barista-quality coffee at home.", "category": "Kitchen"},
    {"name": "Non-Stick Skillet", "description": "Professional grade ceramic coating for easy food release and cleaning.", "category": "Kitchen"},
    
    # Home Office
    {"name": "Ergonomic Chair", "description": "Adjustable lumbar support and breathable mesh back for long work hours.", "category": "Office"},
    {"name": "Standing Desk", "description": "Electric height-adjustable workstation with memory presets.", "category": "Office"}