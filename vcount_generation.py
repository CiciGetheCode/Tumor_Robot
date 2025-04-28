num_faces = 278820
vertices_per_face = 3
i = 0
while i<num_faces:
    # Generate <vcount>
    vcount = "3 " * num_faces  # Repeats "3" num_faces times and adds a space
    vcount = vcount.strip()  # Remove the trailing space
    i +=1
# Print or write to a file
print(f"<vcount>{vcount}</vcount>")
print(i)