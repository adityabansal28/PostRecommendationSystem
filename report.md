# Hybrid Recommender (Content + CF)

**Data**: Users (n=50), Posts (n=100), Engagements (n=1000).

## Method
1. **Content-based**:
   - TF-IDF over `content_type + tags` for each post.
   - User profile = 0.6 × TF-IDF(interests) + 0.4 × TF-IDF(history-weighted).
   - Score = cosine similarity between user profile and post vectors.

2. **Collaborative Filtering**:
   - User–Item matrix from `engagement`.
   - Low-rank factors via TruncatedSVD (k=20).
   - Score = reconstructed dot product between user and item factors.

3. **Hybrid**:
   - Final score = 0.5 × Content + 0.5 × CF.
   - Exclude posts already positively engaged by the user.
   - Return Top-3 per user.

## Quick Validation (simulated)
- 10% random holdout of positive interactions per user (no timestamps available).
- **Hit-Rate@3** = 0.000 (nan if no positives/holdouts).

## Notes & Extensions
- If timestamps exist, do a **time-based split** to avoid leakage.
- Tune weights `alpha` (interests vs history) and `lambda` (content vs CF) by validation.
- Replace TF-IDF with **SentenceTransformers** if post text is richer.
- Replace SVD with **implicit ALS** or **Neural CF** for stronger collaborative signal.
- Add **diversity** constraints (MMR) and **freshness** boosts for production.
