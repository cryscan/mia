use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let shader = r#"
struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, write> output: array<vec3<f32>>;

const MAGIC: u32 = 4u;

@compute @workgroup_size(64, 1, 1)
fn process_vertices(@builtin(global_invocation_id) index: vec3<u32>) {
    let i = index.x;
    if (i < arrayLength(&vertices)) {
        let vertex = vertices[i];
        output[i] = vertex.position + MAGIC * vertex.normal;
    }
}
"#;

    let module = naga::front::wgsl::parse_str(shader)?;
    println!("{}", serde_json::to_string_pretty(&module)?);

    Ok(())
}
