from typing import Tuple, Generator, List
import numpy as np

def global_alignment(seq1, seq2, scoring_function):
    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """

    # Initialize the scoring matrix.
    scoring_matrix = np.zeros((len(seq1) + 1, len(seq2) + 1))

    # Fill the scoring matrix.
    for i in range(1, len(seq1) + 1):
        scoring_matrix[i, 0] = scoring_matrix[i - 1, 0] + scoring_function(seq1[i - 1], "-")
    for j in range(1, len(seq2) + 1):
        scoring_matrix[0, j] = scoring_matrix[0, j - 1] + scoring_function("-", seq2[j - 1])
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            scoring_matrix[i, j] = max(
                scoring_matrix[i - 1, j - 1] + scoring_function(seq1[i - 1], seq2[j - 1]),
                scoring_matrix[i - 1, j] + scoring_function(seq1[i - 1], "-"),
                scoring_matrix[i, j - 1] + scoring_function("-", seq2[j - 1])
            )

    # Traceback.
    i, j = len(seq1), len(seq2)
    aligned_seq1, aligned_seq2 = "", ""
    score = scoring_matrix[i, j]
    while i > 0 or j > 0:
        if i > 0 and j > 0 and scoring_matrix[i, j] == scoring_matrix[i - 1, j - 1] + scoring_function(seq1[i - 1], seq2[j - 1]):
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            i -= 1
            j -= 1
        elif i > 0 and scoring_matrix[i, j] == scoring_matrix[i - 1, j] + scoring_function(seq1[i - 1], "-"):
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            j -= 1

    return aligned_seq1, aligned_seq2, score #scoring_matrix[-1, -1]
    

def local_alignment(seq1, seq2, scoring_function):
    """Local sequence alignment using the Smith-Waterman algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> local_alignment("pending itch", "unending glitch", lambda x, y: [-1, 1][x == y])
    ('ending --itch', 'ending glitch', 9.0)

    Other alignments are not possible.

    """
    # Initialize matrices.
    rows, cols = len(seq1) + 1, len(seq2) + 1
    score_matrix = [[0] * cols for _ in range(rows)]
    traceback_matrix = [[None] * cols for _ in range(rows)]

    max_score = 0
    max_pos = (0, 0)

    # Fill matrices.
    for i in range(1, rows):
        for j in range(1, cols):
            match = score_matrix[i - 1][j - 1] + scoring_function(seq1[i - 1], seq2[j - 1])
            delete = score_matrix[i - 1][j] - 1  # Gap penalty
            insert = score_matrix[i][j - 1] - 1  # Gap penalty
            score_matrix[i][j] = max(0, match, delete, insert)

            # Update the traceback matrix.
            if score_matrix[i][j] == 0:
                traceback_matrix[i][j] = None  # No traceback
            elif score_matrix[i][j] == match:
                traceback_matrix[i][j] = (i - 1, j - 1)
            elif score_matrix[i][j] == delete:
                traceback_matrix[i][j] = (i - 1, j)
            elif score_matrix[i][j] == insert:
                traceback_matrix[i][j] = (i, j - 1)

            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
                max_pos = (i, j)

    # Traceback to get aligned sequences.
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = max_pos

    while traceback_matrix[i][j] is not None:
        prev_i, prev_j = traceback_matrix[i][j]
        if prev_i == i - 1 and prev_j == j - 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
        elif prev_i == i - 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
        elif prev_j == j - 1:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
        i, j = prev_i, prev_j

    # Reverse the sequences since we built them backwards.
    aligned_seq1 = ''.join(reversed(aligned_seq1))
    aligned_seq2 = ''.join(reversed(aligned_seq2))

    return aligned_seq1, aligned_seq2, max_score


def codons(seq: str) -> Generator[str, None, None]:
    """Walk along the string, three nucleotides at a time. Cut off excess."""
    for i in range(0, len(seq) - 2, 3):
        yield seq[i:i + 3]


def translate_to_protein(seq):
    """Translate a nucleotide sequence into a protein sequence.

    Parameters
    ----------
    seq: str

    Returns
    -------
    str
        The translated protein sequence.

    """
    amino_acid_translation_table = {
        "GCT": "A",
        "GCC": "A",
        "GCA": "A",
        "GCG": "A",

        "GAT": "D",
        "GAC": "D",

        "TTT": "F",
        "TTC": "F",
 
        "CAT": "H",
        "CAC": "H",

        "AAA": "K",
        "AAG": "K",

        "ATG": "M",
        
        "CCT": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",

        "CGT": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "AGA": "R",
        "AGG": "R",

        "ACT": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",

        "TGG": "W",

        "TGT": "C",
        "TGC": "C",

        "GAA": "E",
        "GAG": "E",

        "GGT": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",

        "ATT": "I",
        "ATC": "I",
        "ATA": "I",

        "TTA": "L",
        "TTG": "L",
        "CTT": "L",
        "CTC": "L",
        "CTA": "L",
        "CTG": "L",

        "AAT": "N",
        "AAC": "N",

        "CAA": "Q",
        "CAG": "Q",

        "TCT": "S",
        "TCC": "S",
        "TCA": "S",
        "TCG": "S",
        "AGT": "S",
        "AGC": "S",

        "GTT": "V",
        "GTC": "V",
        "GTA": "V",
        "GTG": "V",

        "TAT": "Y",
        "TAC": "Y",

        "TAA": "",
        "TGA": "",
        "TAG": "",
    }

    protein_seq = ""
    for codon in codons(seq):
        protein_seq += amino_acid_translation_table[codon]

    return protein_seq
