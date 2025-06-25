use std::fmt::Write;

/// A builder for constructing WGSL compute shaders piece by piece.
#[derive(Debug, Default, Clone)]
pub struct ShaderBuilder {
    bindings: Vec<Binding>,
    variables: Vec<Variable>,
    functions: Vec<Function>,
    structs: Vec<Struct>,
    constants: Vec<Constant>,
}

#[derive(Debug, Clone)]
pub struct Binding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub binding_type: BindingType,
    pub access: AccessMode,
}

#[derive(Debug, Clone)]
pub enum BindingType {
    Storage(String), // type name
    Uniform(String), // type name
    Texture2d,
    Sampler,
}

#[derive(Debug, Clone, Copy)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub struct LaunchSize {
    pub x: u32,
    pub y: Option<u32>,
    pub z: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub r#type: String,
    pub initial_value: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub launch_size: Option<LaunchSize>,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<String>,
    pub body: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub param_type: String,
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<StructField>,
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub field_type: String,
    pub attributes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Constant {
    pub name: String,
    pub r#type: String,
    pub value: String,
}

impl ShaderBuilder {
    /// Create a new shader builder.
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            variables: Vec::new(),
            functions: Vec::new(),
            structs: Vec::new(),
            constants: Vec::new(),
        }
    }

    /// Add a storage buffer binding.
    pub fn add_storage_buffer(
        mut self,
        group: u32,
        binding: u32,
        name: impl Into<String>,
        r#type: impl Into<String>,
        access: AccessMode,
    ) -> Self {
        self.bindings.push(Binding {
            group,
            binding,
            name: name.into(),
            binding_type: BindingType::Storage(r#type.into()),
            access,
        });
        self
    }

    /// Add a uniform buffer binding.
    pub fn add_uniform_buffer(
        mut self,
        group: u32,
        binding: u32,
        name: impl Into<String>,
        r#type: impl Into<String>,
    ) -> Self {
        self.bindings.push(Binding {
            group,
            binding,
            name: name.into(),
            binding_type: BindingType::Uniform(r#type.into()),
            access: AccessMode::Read,
        });
        self
    }

    /// Add a struct definition.
    pub fn add_struct<F>(mut self, name: impl Into<String>, f: F) -> Self
    where
        F: FnOnce(StructBuilder) -> StructBuilder,
    {
        let r#struct = f(StructBuilder::new(name)).build();
        self.structs.push(r#struct);
        self
    }

    /// Add a constant.
    pub fn add_constant(
        mut self,
        name: impl Into<String>,
        r#type: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.constants.push(Constant {
            name: name.into(),
            r#type: r#type.into(),
            value: value.into(),
        });
        self
    }

    /// Add a variable in workgroup address space.
    pub fn add_variable(mut self, name: impl Into<String>, r#type: impl Into<String>) -> Self {
        self.variables.push(Variable {
            name: name.into(),
            r#type: r#type.into(),
            initial_value: None,
        });
        self
    }

    /// Add a variable with an initial value.
    pub fn add_variable_with_initial_value(
        mut self,
        name: impl Into<String>,
        r#type: impl Into<String>,
        initial_value: impl Into<String>,
    ) -> Self {
        self.variables.push(Variable {
            name: name.into(),
            r#type: r#type.into(),
            initial_value: Some(initial_value.into()),
        });
        self
    }

    pub fn add_function<F>(mut self, name: impl Into<String>, f: F) -> Self
    where
        F: FnOnce(FunctionBuilder) -> FunctionBuilder,
    {
        let function = f(FunctionBuilder::new(name)).build();
        self.functions.push(function);
        self
    }

    /// Generate the complete WGSL shader code.
    pub fn build(self) -> Result<String, std::fmt::Error> {
        let mut shader = String::new();

        // add constants
        for constant in &self.constants {
            writeln!(
                shader,
                "const {}: {} = {};",
                constant.name, constant.r#type, constant.value
            )?;
        }
        if !self.constants.is_empty() {
            writeln!(shader)?;
        }

        // add struct definitions
        for struct_def in &self.structs {
            writeln!(shader, "struct {} {{", struct_def.name)?;
            for field in &struct_def.fields {
                let attrs = match field.attributes.is_empty() {
                    true => String::new(),
                    false => format!("{} ", field.attributes.join(" ")),
                };
                writeln!(shader, "    {}{}: {},", attrs, field.name, field.field_type)?;
            }
            writeln!(shader, "}}")?;
            writeln!(shader)?;
        }

        // add bindings
        for binding in &self.bindings {
            let access_str = match binding.access {
                AccessMode::Read => "read",
                AccessMode::Write => "write",
                AccessMode::ReadWrite => "read_write",
            };

            let binding_str = match &binding.binding_type {
                BindingType::Storage(r#type) => {
                    format!("var<storage, {}> {}: {};", access_str, binding.name, r#type)
                }
                BindingType::Uniform(r#type) => {
                    format!("var<uniform> {}: {};", binding.name, r#type)
                }
                BindingType::Texture2d => {
                    format!("var {}: texture_2d<f32>;", binding.name)
                }
                BindingType::Sampler => {
                    format!("var {}: sampler;", binding.name)
                }
            };

            writeln!(
                shader,
                "@group({}) @binding({}) {}",
                binding.group, binding.binding, binding_str
            )?;
        }
        if !self.bindings.is_empty() {
            writeln!(shader)?;
        }

        // add variables
        for variable in &self.variables {
            match &variable.initial_value {
                Some(initial_value) => {
                    writeln!(
                        shader,
                        "var<workgroup> {}: {} = {};",
                        variable.name, variable.r#type, initial_value
                    )?;
                }
                None => {
                    writeln!(
                        shader,
                        "var<workgroup> {}: {};",
                        variable.name, variable.r#type
                    )?;
                }
            }
        }
        if !self.variables.is_empty() {
            writeln!(shader)?;
        }

        // Add functions
        for function in &self.functions {
            let launch = match &function.launch_size {
                Some(launch_size) => {
                    let x = launch_size.x;
                    let y = launch_size.y.unwrap_or(1);
                    let z = launch_size.z.unwrap_or(1);
                    format!("@compute @workgroup_size({}, {}, {})", x, y, z)
                }
                None => String::new(),
            };

            let params = function
                .parameters
                .iter()
                .map(|p| format!("{}: {}", p.name, p.param_type))
                .collect::<Vec<_>>()
                .join(", ");

            match &function.return_type {
                Some(return_type) => writeln!(
                    shader,
                    "{}\nfn {}({}) -> {} {{",
                    launch, function.name, params, return_type
                )?,
                None => writeln!(shader, "{}\nfn {}({}) {{", launch, function.name, params)?,
            }

            // Add function body with proper indentation
            for line in &function.body {
                if !line.trim().is_empty() {
                    writeln!(shader, "    {}", line)?;
                } else {
                    writeln!(shader)?;
                }
            }
            writeln!(shader, "}}")?;
            writeln!(shader)?;
        }

        Ok(shader)
    }
}

/// Builder for struct definitions.
#[derive(Debug, Clone)]
pub struct StructBuilder {
    name: String,
    fields: Vec<StructField>,
}

impl StructBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let fields = Vec::new();
        Self { name, fields }
    }

    /// Add a field to the struct.
    pub fn add_field(mut self, name: impl Into<String>, field_type: impl Into<String>) -> Self {
        self.fields.push(StructField {
            name: name.into(),
            field_type: field_type.into(),
            attributes: Vec::new(),
        });
        self
    }

    /// Add a field with attributes to the struct.
    pub fn add_field_with_attributes(
        mut self,
        name: impl Into<String>,
        field_type: impl Into<String>,
        attributes: Vec<String>,
    ) -> Self {
        self.fields.push(StructField {
            name: name.into(),
            field_type: field_type.into(),
            attributes,
        });
        self
    }
    /// Finish building the struct.
    pub fn build(self) -> Struct {
        Struct {
            name: self.name,
            fields: self.fields,
        }
    }
}

/// Builder for a code block encapsulated by `{}`.
#[derive(Debug, Clone)]
pub struct BlockBuilder {
    lines: Vec<String>,
    indent: usize,
}

impl BlockBuilder {
    /// Create a new body builder.
    pub fn new(indent: usize) -> Self {
        let lines = Vec::new();
        Self { lines, indent }
    }

    /// Add a line to the body.
    pub fn add_line(mut self, line: impl Into<String>) -> Self {
        let line = line.into();
        let indentation = "    ".repeat(self.indent);
        self.lines.push(format!("{}{}", indentation, line));
        self
    }

    /// Add a block to the body.
    pub fn add_block<F>(mut self, header: impl Into<String>, f: F) -> Self
    where
        F: FnOnce(BlockBuilder) -> BlockBuilder,
    {
        let header = header.into();
        let indentation = "    ".repeat(self.indent);
        self.lines.push(format!("{}{} {{", indentation, header));

        let inner_builder = f(BlockBuilder::new(self.indent + 1));

        for line in inner_builder.lines {
            self.lines.push(format!("{    }{}", indentation, line));
        }

        self.lines.push(format!("{}}}", indentation));
        self
    }

    /// Add an if statement.
    pub fn add_if(
        self,
        condition: impl Into<String>,
        body_fn: impl FnOnce(BlockBuilder) -> BlockBuilder,
    ) -> Self {
        self.add_block(format!("if ({})", condition.into()), body_fn)
    }

    /// Add a for loop.
    pub fn add_for(
        self,
        var: impl Into<String>,
        start: impl Into<String>,
        end: impl Into<String>,
        body_fn: impl FnOnce(BlockBuilder) -> BlockBuilder,
    ) -> Self {
        let var = var.into();
        let header = format!(
            "for (var {}: u32 = {}u; {} < {}; {}++)",
            var,
            start.into(),
            var,
            end.into(),
            var
        );
        self.add_block(header, body_fn)
    }

    /// Add an unrolled loop by generating the body multiple times.
    /// This is useful for small, fixed iteration counts where unrolling can improve performance.
    pub fn add_unrolled_loop<I, F>(
        mut self,
        var: impl Into<String>,
        iter: impl IntoIterator<Item = I>,
        body_fn: F,
    ) -> Self
    where
        F: Fn(&str, I, BlockBuilder) -> BlockBuilder,
    {
        let var_name = var.into();
        for i in iter.into_iter() {
            let block_builder = BlockBuilder::new(self.indent);
            let updated_builder = body_fn(&var_name, i, block_builder);
            for line in updated_builder.lines {
                self.lines.push(line);
            }
        }
        self
    }

    /// Build the block.
    pub fn build(self) -> Vec<String> {
        self.lines
    }
}

#[derive(Debug, Clone)]
pub struct FunctionBuilder {
    name: String,
    launch_size: Option<LaunchSize>,
    parameters: Vec<Parameter>,
    return_type: Option<String>,
    body: BlockBuilder,
}

impl FunctionBuilder {
    /// Create a new function builder.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let launch_size = None;
        let parameters = Vec::new();
        let return_type = None;
        let body = BlockBuilder::new(0);
        Self {
            name,
            launch_size,
            parameters,
            return_type,
            body,
        }
    }

    /// Set the workgroup size.
    pub fn launch_size(mut self, x: u32, y: Option<u32>, z: Option<u32>) -> Self {
        self.launch_size = Some(LaunchSize { x, y, z });
        self
    }

    /// Add a parameter to the function.
    pub fn add_parameter(mut self, name: impl Into<String>, param_type: impl Into<String>) -> Self {
        self.parameters.push(Parameter {
            name: name.into(),
            param_type: param_type.into(),
        });
        self
    }

    /// Set the return type of the function.
    pub fn return_type(mut self, return_type: impl Into<String>) -> Self {
        self.return_type = Some(return_type.into());
        self
    }

    /// Add a line to the function body.
    pub fn add_line(mut self, line: impl Into<String>) -> Self {
        self.body = self.body.add_line(line);
        self
    }

    /// Add an if statement to the function body.
    pub fn add_if(
        mut self,
        condition: impl Into<String>,
        body_fn: impl FnOnce(BlockBuilder) -> BlockBuilder,
    ) -> Self {
        self.body = self.body.add_if(condition, body_fn);
        self
    }

    /// Add a for loop to the function body.
    pub fn add_for(
        mut self,
        var: impl Into<String>,
        start: impl Into<String>,
        end: impl Into<String>,
        body_fn: impl FnOnce(BlockBuilder) -> BlockBuilder,
    ) -> Self {
        self.body = self.body.add_for(var, start, end, body_fn);
        self
    }

    /// Add an unrolled loop to the function body.
    pub fn add_unrolled_loop<I, F>(
        mut self,
        var: impl Into<String>,
        iter: impl IntoIterator<Item = I>,
        body_fn: F,
    ) -> Self
    where
        F: Fn(&str, I, BlockBuilder) -> BlockBuilder,
    {
        self.body = self.body.add_unrolled_loop(var, iter, body_fn);
        self
    }

    /// Finish building the function and return to the shader builder.
    pub fn build(self) -> Function {
        Function {
            name: self.name,
            launch_size: self.launch_size,
            parameters: self.parameters,
            return_type: self.return_type,
            body: self.body.build(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_compute_shader() {
        let shader = ShaderBuilder::new()
            .add_storage_buffer(0, 0, "input", "array<f32>", AccessMode::Read)
            .add_storage_buffer(0, 1, "output", "array<f32>", AccessMode::Write)
            .add_function("main", |f| {
                f.launch_size(64, None, None)
                    .add_parameter("index", "@builtin(global_invocation_id) vec3<u32>")
                    .add_line("let i = index.x;")
                    .add_line("output[i] = input[i] * 2.0;")
            })
            .build()
            .unwrap();

        let expected = r#"@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(index: @builtin(global_invocation_id) vec3<u32>) {
    let i = index.x;
    output[i] = input[i] * 2.0;
}

"#;

        assert_eq!(shader, expected);
    }

    #[test]
    fn test_shader_with_struct_and_uniform() {
        let shader = ShaderBuilder::new()
            .add_struct("Params", |s| {
                s.add_field("scale", "f32").add_field("offset", "f32")
            })
            .add_uniform_buffer(0, 0, "params", "Params")
            .add_storage_buffer(0, 1, "data", "array<f32>", AccessMode::ReadWrite)
            .add_function("compute", |f| {
                f.launch_size(256, None, None)
                    .add_parameter("id", "@builtin(global_invocation_id) vec3<u32>")
                    .add_line("let index = id.x;")
                    .add_if("index < arrayLength(&data)", |b| {
                        b.add_line("data[index] = data[index] * params.scale + params.offset;")
                    })
            })
            .build()
            .unwrap();

        let expected = r#"struct Params {
    scale: f32,
    offset: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn compute(id: @builtin(global_invocation_id) vec3<u32>) {
    let index = id.x;
    if (index < arrayLength(&data)) {
        data[index] = data[index] * params.scale + params.offset;
    }
}

"#;

        assert_eq!(shader, expected);
    }

    #[test]
    fn test_shader_with_constants_and_variables() {
        let shader = ShaderBuilder::new()
            .add_constant("WORKGROUP_SIZE", "u32", "64u")
            .add_constant("PI", "f32", "3.14159265359")
            .add_variable("shared_data", "array<f32, 64>")
            .add_storage_buffer(0, 0, "input", "array<f32>", AccessMode::Read)
            .add_storage_buffer(0, 1, "output", "array<f32>", AccessMode::Write)
            .add_function("main", |f| {
                f.launch_size(64, None, None)
                    .add_parameter("local_id", "@builtin(local_invocation_id) vec3<u32>")
                    .add_parameter("global_id", "@builtin(global_invocation_id) vec3<u32>")
                    .add_line("let lid = local_id.x;")
                    .add_line("let gid = global_id.x;")
                    .add_line("shared_data[lid] = input[gid];")
                    .add_line("workgroupBarrier();")
                    .add_line("output[gid] = shared_data[lid] * PI;")
            })
            .build()
            .unwrap();

        let expected = r#"const WORKGROUP_SIZE: u32 = 64u;
const PI: f32 = 3.14159265359;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, write> output: array<f32>;

var<workgroup> shared_data: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(local_id: @builtin(local_invocation_id) vec3<u32>, global_id: @builtin(global_invocation_id) vec3<u32>) {
    let lid = local_id.x;
    let gid = global_id.x;
    shared_data[lid] = input[gid];
    workgroupBarrier();
    output[gid] = shared_data[lid] * PI;
}

"#;

        assert_eq!(shader, expected);
    }

    #[test]
    fn test_shader_with_for_loop() {
        let shader = ShaderBuilder::new()
            .add_storage_buffer(0, 0, "matrix_a", "array<f32>", AccessMode::Read)
            .add_storage_buffer(0, 1, "matrix_b", "array<f32>", AccessMode::Read)
            .add_storage_buffer(0, 2, "matrix_c", "array<f32>", AccessMode::Write)
            .add_uniform_buffer(0, 3, "size", "u32")
            .add_function("matrix_multiply", |f| {
                f.launch_size(16, Some(16), None)
                    .add_parameter("id", "@builtin(global_invocation_id) vec3<u32>")
                    .add_line("let row = id.y;")
                    .add_line("let col = id.x;")
                    .add_line("var sum = 0.0;")
                    .add_for("k", "0u", "size", |b| {
                        b.add_line("let a_idx = row * size + k;")
                            .add_line("let b_idx = k * size + col;")
                            .add_line("sum += matrix_a[a_idx] * matrix_b[b_idx];")
                    })
                    .add_line("let c_idx = row * size + col;")
                    .add_line("matrix_c[c_idx] = sum;")
            })
            .build()
            .unwrap();

        let expected = r#"@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(16, 16, 1)
fn matrix_multiply(id: @builtin(global_invocation_id) vec3<u32>) {
    let row = id.y;
    let col = id.x;
    var sum = 0.0;
    for (var k: u32 = 0uu; k < size; k++) {
        let a_idx = row * size + k;
        let b_idx = k * size + col;
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }
    let c_idx = row * size + col;
    matrix_c[c_idx] = sum;
}

"#;

        assert_eq!(shader, expected);
    }

    #[test]
    fn test_shader_with_struct_attributes() {
        let shader = ShaderBuilder::new()
            .add_struct("Vertex", |s| {
                s.add_field_with_attributes(
                    "position",
                    "vec3<f32>",
                    vec!["@location(0)".to_string()],
                )
                .add_field_with_attributes("normal", "vec3<f32>", vec!["@location(1)".to_string()])
                .add_field_with_attributes(
                    "uv",
                    "vec2<f32>",
                    vec!["@location(2)".to_string()],
                )
            })
            .add_storage_buffer(0, 0, "vertices", "array<Vertex>", AccessMode::Read)
            .add_storage_buffer(0, 1, "output", "array<vec3<f32>>", AccessMode::Write)
            .add_function("process_vertices", |f| {
                f.launch_size(64, None, None)
                    .add_parameter("index", "@builtin(global_invocation_id) vec3<u32>")
                    .add_line("let i = index.x;")
                    .add_if("i < arrayLength(&vertices)", |b| {
                        b.add_line("let vertex = vertices[i];")
                            .add_line("output[i] = vertex.position + vertex.normal;")
                    })
            })
            .build()
            .unwrap();

        let expected = r#"struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, write> output: array<vec3<f32>>;

@compute @workgroup_size(64, 1, 1)
fn process_vertices(index: @builtin(global_invocation_id) vec3<u32>) {
    let i = index.x;
    if (i < arrayLength(&vertices)) {
        let vertex = vertices[i];
        output[i] = vertex.position + vertex.normal;
    }
}

"#;

        assert_eq!(shader, expected);
    }

    #[test]
    fn test_empty_shader() {
        let shader = ShaderBuilder::new().build().unwrap();
        assert_eq!(shader, "");
    }

    #[test]
    fn test_shader_with_multiple_functions() {
        let shader = ShaderBuilder::new()
            .add_storage_buffer(0, 0, "data", "array<f32>", AccessMode::ReadWrite)
            .add_function("helper", |f| {
                f.add_parameter("value", "f32")
                    .return_type("f32")
                    .add_line("return value * 2.0;")
            })
            .add_function("main", |f| {
                f.launch_size(64, None, None)
                    .add_parameter("id", "@builtin(global_invocation_id) vec3<u32>")
                    .add_line("let index = id.x;")
                    .add_line("data[index] = helper(data[index]);")
            })
            .build()
            .unwrap();

        let expected = r#"@group(0) @binding(0) var<storage, read_write> data: array<f32>;


fn helper(value: f32) -> f32 {
    return value * 2.0;
}

@compute @workgroup_size(64, 1, 1)
fn main(id: @builtin(global_invocation_id) vec3<u32>) {
    let index = id.x;
    data[index] = helper(data[index]);
}

"#;

        assert_eq!(shader, expected);
    }

    #[test]
    fn test_shader_with_unrolled_loop() {
        let shader = ShaderBuilder::new()
            .add_storage_buffer(0, 0, "input", "array<f32>", AccessMode::Read)
            .add_storage_buffer(0, 1, "output", "array<f32>", AccessMode::Write)
            .add_function("vector_dot_product", |f| {
                f.launch_size(64, None, None)
                    .add_parameter("id", "@builtin(global_invocation_id) vec3<u32>")
                    .add_line("let base_idx = id.x * 4u;")
                    .add_line("var sum = 0.0;")
                    .add_unrolled_loop("i", 0..4, |_var, iteration, body| {
                        body.add_line(format!(
                            "sum += input[base_idx + {}u] * input[base_idx + {}u];",
                            iteration, iteration
                        ))
                    })
                    .add_line("output[id.x] = sum;")
            })
            .build()
            .unwrap();

        let expected = r#"@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn vector_dot_product(id: @builtin(global_invocation_id) vec3<u32>) {
    let base_idx = id.x * 4u;
    var sum = 0.0;
    sum += input[base_idx + 0u] * input[base_idx + 0u];
    sum += input[base_idx + 1u] * input[base_idx + 1u];
    sum += input[base_idx + 2u] * input[base_idx + 2u];
    sum += input[base_idx + 3u] * input[base_idx + 3u];
    output[id.x] = sum;
}

"#;

        assert_eq!(shader, expected);
    }

    #[test]
    fn test_shader_with_complex_unrolled_loop() {
        let shader = ShaderBuilder::new()
            .add_storage_buffer(0, 0, "matrix_a", "array<f32>", AccessMode::Read)
            .add_storage_buffer(0, 1, "matrix_b", "array<f32>", AccessMode::Read)
            .add_storage_buffer(0, 2, "result", "array<f32>", AccessMode::Write)
            .add_uniform_buffer(0, 3, "size", "u32")
            .add_function("unrolled_matrix_multiply", |f| {
                f.launch_size(16, Some(16), None)
                    .add_parameter("id", "@builtin(global_invocation_id) vec3<u32>")
                    .add_line("let row = id.y;")
                    .add_line("let col = id.x;")
                    .add_line("var sum = 0.0;")
                    .add_unrolled_loop("k", 0..8, |_var, k, body| {
                        body.add_line(format!("let a_idx_{} = row * size + {}u;", k, k))
                            .add_line(format!("let b_idx_{} = {}u * size + col;", k, k))
                            .add_line(format!(
                                "sum += matrix_a[a_idx_{}] * matrix_b[b_idx_{}];",
                                k, k
                            ))
                    })
                    .add_line("let result_idx = row * size + col;")
                    .add_line("result[result_idx] = sum;")
            })
            .build()
            .unwrap();

        let expected = r#"@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, write> result: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(16, 16, 1)
fn unrolled_matrix_multiply(id: @builtin(global_invocation_id) vec3<u32>) {
    let row = id.y;
    let col = id.x;
    var sum = 0.0;
    let a_idx_0 = row * size + 0u;
    let b_idx_0 = 0u * size + col;
    sum += matrix_a[a_idx_0] * matrix_b[b_idx_0];
    let a_idx_1 = row * size + 1u;
    let b_idx_1 = 1u * size + col;
    sum += matrix_a[a_idx_1] * matrix_b[b_idx_1];
    let a_idx_2 = row * size + 2u;
    let b_idx_2 = 2u * size + col;
    sum += matrix_a[a_idx_2] * matrix_b[b_idx_2];
    let a_idx_3 = row * size + 3u;
    let b_idx_3 = 3u * size + col;
    sum += matrix_a[a_idx_3] * matrix_b[b_idx_3];
    let a_idx_4 = row * size + 4u;
    let b_idx_4 = 4u * size + col;
    sum += matrix_a[a_idx_4] * matrix_b[b_idx_4];
    let a_idx_5 = row * size + 5u;
    let b_idx_5 = 5u * size + col;
    sum += matrix_a[a_idx_5] * matrix_b[b_idx_5];
    let a_idx_6 = row * size + 6u;
    let b_idx_6 = 6u * size + col;
    sum += matrix_a[a_idx_6] * matrix_b[b_idx_6];
    let a_idx_7 = row * size + 7u;
    let b_idx_7 = 7u * size + col;
    sum += matrix_a[a_idx_7] * matrix_b[b_idx_7];
    let result_idx = row * size + col;
    result[result_idx] = sum;
}

"#;

        assert_eq!(shader, expected);
    }
}
