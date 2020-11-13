def defangIPaddr( address):
    return address.replace('.','[.]')

print(defangIPaddr("1.1.1.1"))