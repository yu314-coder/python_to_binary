# -*- coding: utf-8 -*-
"""What a source file may contain: a docstring with ácentos and 中文."""
# A Python name is not an ASCII name, and a C name is. Nothing here goes
# wrong by accident - the emitter never spells a Python name into C - but it
# is the kind of thing that does.
变量 = 5


def 函数(值):
    """Función con nombre no inglés."""
    return 值 * 2


class Σ:
    λ = 3

    def μέθοδος(self):
        return Σ.λ


def función(x):
    return x + 1


print(函数(变量), Σ().μέθοδος(), función(1))
print(函数.__doc__, __doc__)

# Identifiers are normalised to NFKC, so these two are one name.
ﬁle = "ligature"
print(file if False else ﬁle)

emoji = "🐍 python 🎉"
print(emoji, len(emoji), emoji[0], emoji.encode("utf-8")[:4])

# A long chain and a deep nest, which are about the compiler rather than the
# alphabet, but belong to the same question: what does the source look like.
print(sum(range(200)), 0 + 1 + 2 + 3 + 4 + 5)
deep = ((((((((((1))))))))))
print(deep)
