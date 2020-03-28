import glob



for n in glob.iglob("./data/*.txt"):
	W=open(n,"r").read().lower()
	N=""
	for k in W:
		for c in k:
			if (c in "abcdefghijklmnopqrstuvwxyz\n"):
				N+=c
	open(n,"w").write(N)