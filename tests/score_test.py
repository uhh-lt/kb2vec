# inside the link function

cds = []
for i, candidate in enumerate(candidates):
    cds.append((scores[i][0], candidate.name,
                sims.toarray()[i][0], overlap_scores[i][0]))

for c_score, c_name, c_sim, c_overlap in sorted(cds, reverse=True):
    print("- {} {:.2f} {:.2f} {:.2f}".format(c_name, c_score, c_sim, c_overlap))
