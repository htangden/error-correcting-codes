class Mod:
    def __init__(self, a: int, n: int):
        self.a = a%n
        self.n = n

    def __add__(self, other_number):
        if isinstance(other_number, Mod):
            if self.n == other_number.n:
                return Mod((self.a + other_number.a) % self.n, self.n)
        else:
            return False
    
    def __sub__(self, other):
        if isinstance(other, Mod):
            return Mod(self.a - other.a, self.n)
        return False
        
    def __mul__(self, other):
        if isinstance(other, Mod):
            if self.n == other.n:
                return Mod((self.a * other.a) % self.n, self.n)
        else: 
            return False

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Mod):
            inv = pow(other.a, -1, self.n)
            return Mod(self.a * inv, self.n)
        return False
    
    def __neg__(self):
        return Mod(-self.a, self.n)
        
    def __str__(self):
        return f"({self.a} mod {self.n})"
    
    def __eq__(self, other):
        if isinstance(other, Mod):
            return self.a == other.a and self.n == other.n
        return False

    def __repr__(self):
        return f"({self.a} mod {self.n})"
    
    

if __name__ == "__main__":
    breakpoint()

 