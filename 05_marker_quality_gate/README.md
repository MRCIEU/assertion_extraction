# Step 05 — Marker quality gate

Verifies offset-based entity marker insertion and rebuilds train caches.

Method: Compare native-offset insertion against prior string-match insertion on training, benchmark, and pool evaluation paths.

| Check | Result |
| --- | ---: |
| Offset gate | passed |
| Training offset insertion | 100% |
| Downstream caches rebuilt | steps 10, 11, 20 |
