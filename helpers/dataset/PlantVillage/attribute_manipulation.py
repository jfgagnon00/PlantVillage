def attribute_prettify(species, disease):
    """
    Manipule species et disease pour avoir des une chaine de caracteres
    plus conviviale (disease sont parfoit tres long)
    """
    disease = disease.replace(species, "")
    disease = disease.split("_")
    count = len(disease)
    if count > 2:
        count //= 2
        disease0 = " ".join(disease[:count])
        disease1 = " ".join(disease[count:])
        disease = f"{disease0}\n{disease1}"
    else:
        disease = " ".join(disease)

    return disease