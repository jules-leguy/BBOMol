from chemdesc import ShinglesVectDesc

desc = ShinglesVectDesc(external_desc_id_dict={"N#C": 0}, vect_size=10)

print(desc.fit_transform(["CNN", "C(O)#N"]))
