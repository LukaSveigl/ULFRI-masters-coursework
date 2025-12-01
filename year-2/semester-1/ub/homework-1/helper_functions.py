from typing import Tuple, Generator, List

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def codons(seq: str) -> Generator[str, None, None]:
    """Walk along the string, three nucleotides at a time. Cut off excess."""
    for i in range(0, len(seq) - 2, 3):
        yield seq[i:i + 3]


def extract_gt_orfs(record, start_codons, stop_codons, validate_cds=True, verbose=False):
    """Extract the ground truth ORFs as indicated by the NCBI annotator in the
    gene coding regions (CDS regins) of the genome.

    Parameters
    ----------
    record: SeqRecord
    start_codons: List[str]
    stop_codons: List[str]
    validate_cds: bool
        Filter out NCBI provided ORFs that do not fit our ORF criteria.
    verbose: bool

    Returns
    -------
    List[Tuple[int, int, int]]
        tuples of form (strand, start_loc, stop_loc). Strand should be either 1
        for reference strand and -1 for reverse complement.

    """
    cds_regions = [f for f in record.features if f.type == "CDS"]

    orfs = []
    for region in cds_regions:
        loc = region.location
        seq = record.seq[loc.start.position:loc.end.position]
        if region.strand == -1:
            seq = seq.reverse_complement()
            
        if not validate_cds:
            orfs.append((region.strand, loc.start.position, loc.end.position))
            continue

        try:
            assert seq[:3] in start_codons, "Start codon not found!"
            assert seq[-3:] in stop_codons, "Stop codon not found!"
            # Make sure there are no stop codons in the middle of the sequence
            for codon in codons(seq[3:-3]):
                assert (
                    codon not in stop_codons
                ), f"Stop codon {codon} found in the middle of the sequence!"

            # The CDS looks fine, add it to the ORFs
            orfs.append((region.strand, loc.start.position, loc.end.position))

        except AssertionError as ex:
            if verbose:
                print(
                    "Skipped CDS at region [%d - %d] on strand %d"
                    % (loc.start.position, loc.end.position, region.strand)
                )
                print("\t", str(ex))
                
    # Some ORFs in paramecium have lenghts not divisible by 3. Remove these
    orfs = [orf for orf in orfs if (orf[2] - orf[1]) % 3 == 0]

    return orfs


def find_orfs(sequence, start_codons, stop_codons):
    """Find possible ORF candidates in a single reading frame.

    Parameters
    ----------
    sequence: Seq
    start_codons: List[str]
    stop_codons: List[str]

    Returns
    -------
    List[Tuple[int, int]]
        tuples of form (start_loc, stop_loc)

    """
    orf_candidates = []
    start_of_orf = None
    end_of_orf = None

    codons_list = codons(sequence)
    for index, codon in enumerate(codons_list):
        if codon in start_codons:
            if start_of_orf is None:
                start_of_orf = index * 3
        elif codon in stop_codons:
            if start_of_orf is not None:
                end_of_orf = index * 3 + 3
                orf_candidates.append((start_of_orf, end_of_orf))
                start_of_orf = None
                end_of_orf = None
    
    return orf_candidates


def find_all_orfs(sequence, start_codons, stop_codons):
    """Find ALL the possible ORF candidates in the sequence using all six
    reading frames.

    Parameters
    ----------
    sequence: Seq
    start_codons: List[str]
    stop_codons: List[str]

    Returns
    -------
    List[Tuple[int, int, int]]
        tuples of form (strand, start_loc, stop_loc). Strand should be either 1
        for reference strand and -1 for reverse complement.

    """
    combined_orf_candidates = []

    for i in range(3):
        orf_candidates = find_orfs(sequence[i:], start_codons, stop_codons)
        combined_orf_candidates.extend([(1, start + i, end + i) for start, end in orf_candidates])

    for i in range(3):
        orf_candidates = find_orfs(sequence.reverse_complement()[i:], start_codons, stop_codons)
        combined_orf_candidates.extend([(-1, start + i, end + i) for start, end in orf_candidates])

    return combined_orf_candidates


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


def find_all_orfs_nested(sequence, start_codons, stop_codons):
    """Bonus problem: Find ALL the possible ORF candidates in the sequence using
    the updated definition of ORFs.

    Parameters
    ----------
    sequence: Seq
    start_codons: List[str]
    stop_codons: List[str]

    Returns
    -------
    List[Tuple[int, int, int]]
        tuples of form (strand, start_loc, stop_loc). Strand should be either 1
        for reference strand and -1 for reverse complement.

    """
    all_orf_candidates = []

    for i in range(3):
        orf_candidates = []
        orf_starts = []
        for index, codon in enumerate(codons(sequence[i:])):
            if codon in start_codons:
                orf_starts.append(index * 3)
            elif codon in stop_codons:
                for start in orf_starts:
                    orf_candidates.append((start, index * 3 + 3))
                orf_starts = []

        all_orf_candidates.extend([(1, start + i, end + i) for start, end in orf_candidates])

    sequence = sequence.reverse_complement()

    for i in range(3):
        orf_candidates = []
        orf_starts = []
        for index, codon in enumerate(codons(sequence[i:])):
            if codon in start_codons:
                orf_starts.append(index * 3)
            elif codon in stop_codons:
                for start in orf_starts:
                    orf_candidates.append((start, index * 3 + 3))
                orf_starts = []

        all_orf_candidates.extend([(-1, start + i, end + i) for start, end in orf_candidates])

    return all_orf_candidates
