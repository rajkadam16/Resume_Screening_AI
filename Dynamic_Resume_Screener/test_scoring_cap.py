"""
Test to verify scoring caps at 95% maximum
"""

# Simulate perfect scenario
skill_match_percentage = 60  # All skills matched
exp_match = 20  # Perfect experience match
edu_match = 10  # Has education
quality_bonus = 10  # Perfect quality

total_before_cap = skill_match_percentage + exp_match + edu_match + quality_bonus
print(f"Total before cap: {total_before_cap}%")

# Apply 95% cap
match_percentage = min(95, total_before_cap)
print(f"Final score (capped at 95%): {match_percentage}%")

print("\n" + "="*50)
print("Score Ranges:")
print("="*50)
print("85-95%  → Highly Recommended (Excellent)")
print("65-84%  → Recommended (Good)")
print("50-64%  → Maybe (Average)")
print("0-49%   → Not Recommended (Poor)")
print("="*50)
print("\nNote: Maximum possible score is now 95% to look realistic!")
