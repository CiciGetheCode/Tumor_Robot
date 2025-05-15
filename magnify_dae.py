import xml.etree.ElementTree as ET

def scale_dae_positions(input_path, output_path, scale_factor=5.0):
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    ET.register_namespace('', ns['c'])

    tree = ET.parse(input_path)
    root = tree.getroot()

    for float_array in root.findall(".//c:float_array", ns):
        if 'positions' in float_array.attrib.get('id', ''):
            print(f"Scaling float_array: {float_array.attrib['id']}")
            raw_text = float_array.text.strip()
            numbers = list(map(float, raw_text.split()))
            scaled_numbers = [str(num * scale_factor) for num in numbers]
            float_array.text = ' '.join(scaled_numbers)

    tree.write(output_path)
    print(f"✅ Scaled DAE saved to: {output_path}")

# Example usage
scale_dae_positions(
    input_path="mesh_mask.dae",
    output_path="mesh_mask_scaled_4x.dae",
    scale_factor=4.0
)
