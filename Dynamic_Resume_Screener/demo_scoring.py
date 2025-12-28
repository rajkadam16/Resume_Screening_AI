"""
Visual demonstration of the new scoring system
"""

print("=" * 60)
print("  RESUME SCORING SYSTEM - REALISTIC CAPS")
print("=" * 60)
print()

# Example scenarios
scenarios = [
    {
        "name": "Perfect Match Resume",
        "skills": 60,
        "experience": 20,
        "education": 10,
        "quality": 10,
        "description": "All requirements met perfectly"
    },
    {
        "name": "Very Good Resume",
        "skills": 50,
        "experience": 15,
        "education": 10,
        "quality": 8,
        "description": "Most requirements met"
    },
    {
        "name": "Good Resume",
        "skills": 40,
        "experience": 10,
        "education": 5,
        "quality": 7,
        "description": "Many requirements met"
    },
    {
        "name": "Average Resume",
        "skills": 30,
        "experience": 5,
        "education": 5,
        "quality": 5,
        "description": "Some requirements met"
    },
    {
        "name": "Below Average Resume",
        "skills": 20,
        "experience": 0,
        "education": 0,
        "quality": 4,
        "description": "Few requirements met"
    }
]

for scenario in scenarios:
    total = scenario["skills"] + scenario["experience"] + scenario["education"] + scenario["quality"]
    capped = min(95, total)
    
    # Determine recommendation
    if capped >= 80:
        recommendation = "Highly Recommended ⭐"
        color = "🟢"
    elif capped >= 65:
        recommendation = "Recommended ✓"
        color = "🔵"
    elif capped >= 50:
        recommendation = "Maybe ?"
        color = "🟠"
    else:
        recommendation = "Not Recommended ✗"
        color = "🔴"
    
    print(f"{color} {scenario['name']}")
    print(f"   {scenario['description']}")
    print(f"   Skills: {scenario['skills']} | Exp: {scenario['experience']} | Edu: {scenario['education']} | Quality: {scenario['quality']}")
    print(f"   Raw Total: {total}% → Final Score: {capped}%")
    print(f"   → {recommendation}")
    print()

print("=" * 60)
print("KEY IMPROVEMENTS:")
print("=" * 60)
print("✓ Maximum score is now 95% (not 100%)")
print("✓ More realistic and believable scores")
print("✓ Better differentiation between candidates")
print("✓ Reflects real-world evaluation standards")
print("=" * 60)
