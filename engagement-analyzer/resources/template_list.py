TPL_ENT = """
<mark class="entity" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    {text}
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">{label}</span>
</mark>
"""

TPL_SPANS = """
<div class="spans" style="line-height: 4.5;">
    {text}
    {span_slices}
    {span_starts}
</div>
"""

TPL_SPAN = """
<span style="font-weight: bold; display: inline-block; line-height: 3; padding-bottom: 12px;position: relative;">
    {text}
    {span_slices}
    {span_starts}
</span>
"""

TPL_SPAN_SLICE = """
<span style="background: {bg}; top: {top_offset}px;  display: inline-block; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
"""

TPL_SPAN_START = """
<span style="background: {bg}; top: {top_offset}px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
    <span style="background: {bg}; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
       
        {label}{kb_link}
    </span>
</span>

"""

TPL_SPAN_START_RTL = """
<span style="background: {bg}; top: {top_offset}px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
    <span style="background: {bg}; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
        {label}{kb_link}
    </span>
</span>
"""

DEFAULT_TEXT = """Tickner said regardless of the result, the royal commission was a waste of money and he would proceed with a separate inquiry into the issue headed by Justice Jane Matthews. His attack came as the Aboriginal women involved in the case demanded a female minister examine the religious beliefs they claim are inherent in their fight against a bridge to the island near Goolwa in South Australia."""

