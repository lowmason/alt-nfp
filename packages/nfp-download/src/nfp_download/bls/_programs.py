'''
BLS LABSTAT program registry for CE, SM, and EN.

Encodes series ID formats as structured data so that series IDs can be
built, parsed, and validated programmatically.

Ported from eco-stats (https://github.com/lowmason/eco-stats),
retaining only the three programs needed for NFP nowcasting:
- CE: Current Employment Statistics (National)
- SM: State and Area Employment, Hours, and Earnings
- EN: Quarterly Census of Employment and Wages
'''

from __future__ import annotations


class SeriesField:
    '''
    A single positional field within a BLS series ID.

    Positions are 1-indexed to match the official BLS documentation.

    Parameters
    ----------
    name : str
        Machine-readable field name (e.g., ``'seasonal'``).
    start : int
        Start position, 1-indexed inclusive.
    end : int
        End position, 1-indexed inclusive.
    description : str
        Human-readable description of the field.
    '''

    def __init__(
        self,
        name: str,
        start: int,
        end: int,
        description: str = '',
    ) -> None:
        self.name = name
        self.start = start
        self.end = end
        self.description = description

    @property
    def length(self) -> int:
        '''Number of characters this field occupies.'''
        return self.end - self.start + 1

    def extract(self, series_id: str) -> str:
        '''Extract this field's value from a full series ID string.'''
        return series_id[self.start - 1 : self.end]  # noqa: E203

    def __repr__(self) -> str:
        return (
            f"SeriesField({self.name!r}, {self.start}, {self.end}, "
            f"{self.description!r})"
        )


class BLSProgram:
    '''
    Metadata for a single BLS LABSTAT program (survey).

    Parameters
    ----------
    prefix : str
        Two-letter program identifier (e.g., ``'CE'``).
    name : str
        Full program name.
    description : str
        Brief description of the survey.
    fields : list[SeriesField]
        Ordered list of field definitions that make up the series ID format.
    mapping_files : list[str] or None
        Names of available mapping/lookup files.
    '''

    def __init__(
        self,
        prefix: str,
        name: str,
        description: str,
        fields: list[SeriesField],
        mapping_files: list[str] | None = None,
    ) -> None:
        self.prefix = prefix.upper()
        self.name = name
        self.description = description
        self.fields = fields
        self.mapping_files = mapping_files or []

    @property
    def series_id_length(self) -> int:
        '''Expected total length of a series ID for this program.'''
        if not self.fields:
            return 0
        return max(f.end for f in self.fields)

    def field_names(self) -> list[str]:
        '''Return the ordered list of field names.'''
        return [f.name for f in self.fields]

    def get_field(self, name: str) -> SeriesField | None:
        '''Look up a field by name, or return ``None``.'''
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def __repr__(self) -> str:
        return f"BLSProgram({self.prefix!r}, {self.name!r})"


# ---------------------------------------------------------------------------
# Program registry — CE, SM, EN only
# ---------------------------------------------------------------------------

PROGRAMS: dict[str, BLSProgram] = {}


def _register(program: BLSProgram) -> BLSProgram:
    '''Add a program to the global registry and return it.'''
    PROGRAMS[program.prefix] = program
    return program


# -- CE: Current Employment Statistics (National) ---------------------------
_register(
    BLSProgram(
        prefix='CE',
        name='Current Employment Statistics (National)',
        description=(
            'Monthly estimates of employment, hours, and earnings from '
            'the payroll survey (NAICS basis).'
        ),
        fields=[
            SeriesField('prefix', 1, 2, 'Survey prefix (CE)'),
            SeriesField('seasonal', 3, 3, 'Seasonal adjustment code'),
            SeriesField('supersector', 4, 5, 'Supersector code'),
            SeriesField('industry', 6, 11, 'Industry code'),
            SeriesField('data_type', 12, 13, 'Data type code'),
        ],
        mapping_files=[
            'datatype',
            'industry',
            'seasonal',
            'series',
            'supersector',
        ],
    )
)

# -- SM: State and Area Employment, Hours, and Earnings ---------------------
_register(
    BLSProgram(
        prefix='SM',
        name='State and Area Employment, Hours, and Earnings',
        description=(
            'Monthly estimates of employment, hours, and earnings for '
            'states and metropolitan areas (NAICS basis).'
        ),
        fields=[
            SeriesField('prefix', 1, 2, 'Survey prefix (SM)'),
            SeriesField('seasonal', 3, 3, 'Seasonal adjustment code'),
            SeriesField('state', 4, 5, 'State code'),
            SeriesField('area', 6, 10, 'Area code'),
            SeriesField('supersector_industry', 11, 18, 'Supersector/industry code'),
            SeriesField('data_type', 19, 20, 'Data type code'),
        ],
        mapping_files=[
            'area',
            'datatype',
            'industry',
            'seasonal',
            'series',
            'state',
            'supersector',
        ],
    )
)

# -- EN: Quarterly Census of Employment and Wages ---------------------------
_register(
    BLSProgram(
        prefix='EN',
        name='Quarterly Census of Employment and Wages',
        description=(
            'Quarterly employment and wages data covering nearly all '
            'employers, derived from unemployment insurance records.'
        ),
        fields=[
            SeriesField('prefix', 1, 2, 'Survey prefix (EN)'),
            SeriesField('seasonal', 3, 3, 'Seasonal adjustment code'),
            SeriesField('area', 4, 8, 'Area code'),
            SeriesField('data_type', 9, 9, 'Data type code'),
            SeriesField('size', 10, 10, 'Size code'),
            SeriesField('ownership', 11, 11, 'Ownership code'),
            SeriesField('industry', 12, 17, 'Industry code'),
        ],
        mapping_files=[
            'area',
            'datatype',
            'industry',
            'ownership',
            'seasonal',
            'series',
            'size',
        ],
    )
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_program(prefix: str) -> BLSProgram:
    '''
    Look up a BLS program by its two-letter prefix.

    Parameters
    ----------
    prefix : str
        Two-letter program code (case-insensitive).

    Returns
    -------
    BLSProgram

    Raises
    ------
    KeyError
        If the prefix is not in the registry.
    '''
    key = prefix.upper()
    if key not in PROGRAMS:
        available = ', '.join(sorted(PROGRAMS.keys()))
        raise KeyError(
            f"Unknown BLS program prefix {key!r}. "
            f"Available programs: {available}"
        )
    return PROGRAMS[key]


def list_programs() -> dict[str, str]:
    '''
    Return a mapping of all registered program prefixes to their names.

    Returns
    -------
    dict[str, str]
        ``{prefix: program_name}`` for CE, SM, EN.
    '''
    return {p.prefix: p.name for p in PROGRAMS.values()}


# ---------------------------------------------------------------------------
# Series ID builder and parser
# ---------------------------------------------------------------------------


def parse_series_id(series_id: str) -> dict[str, str]:
    '''
    Decompose a BLS series ID into its component fields.

    The first two characters identify the program, which determines
    how the remaining positions are interpreted.

    Parameters
    ----------
    series_id : str
        A full BLS series ID string (e.g., ``'CES0000000001'``).

    Returns
    -------
    dict[str, str]
        Dictionary mapping field names to their extracted values.
        Always includes a ``'program'`` key with the two-letter prefix.

    Raises
    ------
    KeyError
        If the program prefix is not in the registry.
    ValueError
        If the series ID is shorter than the program format requires.
    '''
    if len(series_id) < 2:
        raise ValueError(
            f'Series ID must be at least 2 characters, got {series_id!r}'
        )

    prefix = series_id[:2].upper()
    program = get_program(prefix)

    if len(series_id) < program.series_id_length:
        raise ValueError(
            f'Series ID {series_id!r} is too short for program {prefix}. '
            f'Expected at least {program.series_id_length} characters, '
            f'got {len(series_id)}.'
        )

    result: dict[str, str] = {'program': prefix}
    for field in program.fields:
        result[field.name] = field.extract(series_id)

    return result


def build_series_id(program: str, **components: str) -> str:
    '''
    Construct a BLS series ID from named components.

    Components that are not provided will be filled with ``'0'`` padding
    to the correct width. The ``'prefix'`` component is set automatically
    from the *program* argument.

    Parameters
    ----------
    program : str
        Two-letter program prefix (e.g., ``'CE'``).
    **components : str
        Field values keyed by name.

    Returns
    -------
    str
        The assembled series ID string.

    Raises
    ------
    KeyError
        If the program prefix is not in the registry.
    '''
    prog = get_program(program)

    length = prog.series_id_length
    chars = ['0'] * length

    # Always set the prefix.
    components['prefix'] = prog.prefix

    for field in prog.fields:
        value = components.get(field.name, None)
        if value is not None:
            padded = value.ljust(field.length, '0')[: field.length]
            for i, ch in enumerate(padded):
                chars[field.start - 1 + i] = ch

    return ''.join(chars)
