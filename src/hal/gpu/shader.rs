//! # GPU Shader Builder
//!
//! This module provides a flexible builder pattern for constructing WGSL (WebGPU Shading Language) compute shaders piece by piece. The `ShaderBuilder` is designed to be a foundation for procedural macro generation that can transform Rust-like code into shader builder chains.
//!
//! ## Overview
//!
//! The `ShaderBuilder` allows you to:
//!
//! - **Compose shaders incrementally**: Add bindings, structs, functions, and main body code step by step
//! - **Support runtime generation**: Perfect for loop unrolling, conditional compilation, and dynamic shader variants
//! - **Maintain type safety**: Structured approach to shader construction with clear separation of concerns
//! - **Enable proc macro integration**: Designed as a target for procedural macros that can generate builder chains
//!
//! ## Key Components
//!
//! ### ShaderBuilder
//! The main builder that orchestrates shader construction:
//!
//! ```rust
//! use mia::hal::gpu::shader::{ShaderBuilder, AccessMode};
//!
//! let shader = ShaderBuilder::new()
//!     .workgroup_size(64, None, None)
//!     .add_storage_buffer(0, 0, "input", "array<f32>", AccessMode::Read)
//!     .add_storage_buffer(0, 1, "output", "array<f32>", AccessMode::Write)
//!     .add_main_body_line("let index = global_id.x;")
//!     .add_main_body_line("output[index] = input[index] * 2.0;")
//!     .build();
//! ```
//!
//! ### StructBuilder
//! For defining custom WGSL structs:
//!
//! ```rust
//! use mia::hal::gpu::shader::ShaderBuilder;
//!
//! let shader = ShaderBuilder::new()
//!     .add_struct("Particle")
//!         .add_field("position", "vec3<f32>")
//!         .add_field("velocity", "vec3<f32>")
//!         .add_field_with_attributes("mass", "f32", vec!["@align(16)".to_string()])
//!         .finish_struct()
//!     .build();
//! ```
//!
//! ### FunctionBuilder
//! For adding custom functions:
//!
//! ```rust
//! use mia::hal::gpu::shader::ShaderBuilder;
//!
//! let shader = ShaderBuilder::new()
//!     .add_function("calculate_distance")
//!         .add_parameter("a", "vec3<f32>")
//!         .add_parameter("b", "vec3<f32>")
//!         .return_type("f32")
//!         .body("return length(a - b);")
//!         .finish_function()
//!     .build();
//! ```
//!
//! ## Features
//!
//! ### Binding Management
//! - **Storage buffers**: Read, write, or read-write access modes
//! - **Uniform buffers**: Read-only structured data
//! - **Textures and samplers**: 2D texture support
//! - **Automatic binding generation**: Proper WGSL binding syntax
//!
//! ### Control Flow
//! - **For loops**: Automatic loop generation with proper WGSL syntax
//! - **Conditional statements**: If/else block generation
//! - **Custom code blocks**: Direct WGSL code insertion
//!
//! ### Type System
//! - **Struct definitions**: Custom data structures with attributes
//! - **Constants**: Compile-time constant values
//! - **Variables**: Local and storage class variables
//! - **Function parameters**: Typed parameter lists
//!
//! ### Code Generation
//! - **Proper indentation**: Clean, readable output
//! - **WGSL compliance**: Valid shader syntax
//! - **Workgroup configuration**: Flexible workgroup size specification
//! - **Attribute handling**: Support for WGSL attributes like `@align`, `@size`, etc.
//!
//! ## Usage Patterns
//!
//! ### Simple Compute Shader
//! ```rust
//! use mia::hal::gpu::shader::{ShaderBuilder, AccessMode};
//!
//! fn create_vector_add_shader(size: u32) -> String {
//!     ShaderBuilder::new()
//!         .workgroup_size(64, None, None)
//!         .add_storage_buffer(0, 0, "a", "array<f32>", AccessMode::Read)
//!         .add_storage_buffer(0, 1, "b", "array<f32>", AccessMode::Read)
//!         .add_storage_buffer(0, 2, "result", "array<f32>", AccessMode::Write)
//!         .add_main_body_line("let index = global_id.x;")
//!         .add_main_body_line(format!("if (index >= {}u) {{ return; }}", size))
//!         .add_main_body_line("result[index] = a[index] + b[index];")
//!         .build()
//! }
//! ```
//!
//! ### Loop Unrolling Example
//! ```rust
//! use mia::hal::gpu::shader::{ShaderBuilder, AccessMode};
//!
//! fn create_unrolled_shader(iterations: u32) -> String {
//!     let mut builder = ShaderBuilder::new()
//!         .workgroup_size(1, None, None)
//!         .add_storage_buffer(0, 0, "data", "array<f32>", AccessMode::ReadWrite);
//!     
//!     // unroll loop at compile time
//!     for i in 0..iterations {
//!         builder = builder.add_main_body_line(
//!             format!("data[{}] = data[{}] * 2.0;", i, i)
//!         );
//!     }
//!     
//!     builder.build()
//! }
//! ```
//!
//! ### Complex Shader with Custom Functions
//! ```rust
//! use mia::hal::gpu::shader::{ShaderBuilder, AccessMode};
//!
//! fn create_physics_shader() -> String {
//!     ShaderBuilder::new()
//!         .workgroup_size(256, None, None)
//!         .add_struct("Particle")
//!             .add_field("position", "vec3<f32>")
//!             .add_field("velocity", "vec3<f32>")
//!             .add_field("mass", "f32")
//!             .finish_struct()
//!         .add_storage_buffer(0, 0, "particles", "array<Particle>", AccessMode::ReadWrite)
//!         .add_uniform_buffer(0, 1, "params", "PhysicsParams")
//!         .add_function("apply_force")
//!             .add_parameter("particle", "ptr<storage, Particle, read_write>")
//!             .add_parameter("force", "vec3<f32>")
//!             .add_parameter("dt", "f32")
//!             .body("(*particle).velocity += force / (*particle).mass * dt;")
//!             .finish_function()
//!         .add_main_body_line("let index = global_id.x;")
//!         .add_main_body_line("if (index >= arrayLength(&particles)) { return; }")
//!         .add_main_body_line("apply_force(&particles[index], params.gravity, params.delta_time);")
//!         .build()
//! }
//! ```
//!
//! ## Design Principles
//!
//! ### Builder Pattern
//! The fluent interface allows for readable, chainable shader construction that mirrors the structure of the final WGSL code.
//!
//! ### Proc Macro Ready
//! The builder is designed to be the target of procedural macros that can:
//! - Parse Rust-like syntax
//! - Generate appropriate builder method calls
//! - Handle compile-time optimizations like loop unrolling
//! - Provide type checking and validation
//!
//! ### Runtime Flexibility
//! Shaders can be generated at runtime based on:
//! - Input data characteristics
//! - Performance requirements
//! - Hardware capabilities
//! - User preferences
//!
//! ### Extensibility
//! The modular design allows for easy extension with:
//! - New binding types
//! - Additional control flow constructs
//! - Custom code generation strategies
//! - Platform-specific optimizations
//!
//! ## Future Enhancements
//!
//! - **Vertex and Fragment shaders**: Extend beyond compute shaders
//! - **Advanced control flow**: While loops, switch statements
//! - **Optimization passes**: Dead code elimination, constant folding
//! - **Validation**: Compile-time WGSL validation
//! - **Templates**: Reusable shader components
//! - **Debugging support**: Source maps and debug information

use std::fmt::Write;

/// A builder for constructing WGSL compute shaders piece by piece.
#[derive(Debug, Clone)]
pub struct ShaderBuilder {
    bindings: Vec<Binding>,
    workgroup_size: Option<WorkgroupSize>,
    variables: Vec<Variable>,
    functions: Vec<Function>,
    main_body: Vec<String>,
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
    StorageBuffer(String), // type name
    UniformBuffer(String), // type name
    Texture2d,
    Sampler,
}

#[derive(Debug, Clone)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: Option<u32>,
    pub z: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub var_type: String,
    pub storage_class: Option<String>,
    pub initial_value: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<String>,
    pub body: String,
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
    pub const_type: String,
    pub value: String,
}

impl ShaderBuilder {
    /// Create a new shader builder.
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            workgroup_size: None,
            variables: Vec::new(),
            functions: Vec::new(),
            main_body: Vec::new(),
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
        buffer_type: impl Into<String>,
        access: AccessMode,
    ) -> Self {
        self.bindings.push(Binding {
            group,
            binding,
            name: name.into(),
            binding_type: BindingType::StorageBuffer(buffer_type.into()),
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
        buffer_type: impl Into<String>,
    ) -> Self {
        self.bindings.push(Binding {
            group,
            binding,
            name: name.into(),
            binding_type: BindingType::UniformBuffer(buffer_type.into()),
            access: AccessMode::Read,
        });
        self
    }

    /// Set the workgroup size.
    pub fn workgroup_size(mut self, x: u32, y: Option<u32>, z: Option<u32>) -> Self {
        self.workgroup_size = Some(WorkgroupSize { x, y, z });
        self
    }

    /// Add a struct definition.
    pub fn add_struct(self, name: impl Into<String>) -> StructBuilder {
        StructBuilder {
            shader_builder: self,
            struct_name: name.into(),
            fields: Vec::new(),
        }
    }

    /// Add a constant.
    pub fn add_constant(
        mut self,
        name: impl Into<String>,
        const_type: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.constants.push(Constant {
            name: name.into(),
            const_type: const_type.into(),
            value: value.into(),
        });
        self
    }

    /// Add a variable.
    pub fn add_variable(mut self, name: impl Into<String>, var_type: impl Into<String>) -> Self {
        self.variables.push(Variable {
            name: name.into(),
            var_type: var_type.into(),
            storage_class: None,
            initial_value: None,
        });
        self
    }

    /// Add a variable with storage class.
    pub fn add_variable_with_storage(
        mut self,
        name: impl Into<String>,
        var_type: impl Into<String>,
        storage_class: impl Into<String>,
    ) -> Self {
        self.variables.push(Variable {
            name: name.into(),
            var_type: var_type.into(),
            storage_class: Some(storage_class.into()),
            initial_value: None,
        });
        self
    }

    /// Add a function.
    pub fn add_function(self, name: impl Into<String>) -> FunctionBuilder {
        FunctionBuilder {
            shader_builder: self,
            function_name: name.into(),
            parameters: Vec::new(),
            return_type: None,
            body: String::new(),
        }
    }

    /// Add code to the main compute function body.
    pub fn add_main_body_line(mut self, line: impl Into<String>) -> Self {
        self.main_body.push(line.into());
        self
    }

    /// Add multiple lines to the main compute function body.
    pub fn add_main_body_lines(mut self, lines: Vec<String>) -> Self {
        self.main_body.extend(lines);
        self
    }

    /// Add a for loop to the main body.
    pub fn add_for_loop(
        mut self,
        variable: impl Into<String>,
        start: impl Into<String>,
        end: impl Into<String>,
        body: Vec<String>,
    ) -> Self {
        let var = variable.into();
        let start_val = start.into();
        let end_val = end.into();

        self.main_body.push(format!(
            "for (var {}: u32 = {}u; {} < {}; {}++) {{",
            var, start_val, var, end_val, var
        ));
        for line in body {
            self.main_body.push(format!("    {}", line));
        }
        self.main_body.push("}".to_string());
        self
    }

    /// Add an if statement to the main body.
    pub fn add_if_statement(mut self, condition: impl Into<String>, body: Vec<String>) -> Self {
        self.main_body.push(format!("if ({}) {{", condition.into()));
        for line in body {
            self.main_body.push(format!("    {}", line));
        }
        self.main_body.push("}".to_string());
        self
    }

    /// Generate the complete WGSL shader code.
    pub fn build(self) -> String {
        let mut shader = String::new();

        // Add constants
        for constant in &self.constants {
            writeln!(
                shader,
                "const {}: {} = {};",
                constant.name, constant.const_type, constant.value
            )
            .unwrap();
        }
        if !self.constants.is_empty() {
            writeln!(shader).unwrap();
        }

        // Add struct definitions
        for struct_def in &self.structs {
            writeln!(shader, "struct {} {{", struct_def.name).unwrap();
            for field in &struct_def.fields {
                let attrs = if field.attributes.is_empty() {
                    String::new()
                } else {
                    format!("{} ", field.attributes.join(" "))
                };
                writeln!(shader, "    {}{}: {},", attrs, field.name, field.field_type).unwrap();
            }
            writeln!(shader, "}}").unwrap();
            writeln!(shader).unwrap();
        }

        // Add bindings
        for binding in &self.bindings {
            let access_str = match binding.access {
                AccessMode::Read => "read",
                AccessMode::Write => "write",
                AccessMode::ReadWrite => "read_write",
            };

            let binding_str = match &binding.binding_type {
                BindingType::StorageBuffer(type_name) => {
                    format!(
                        "var<storage, {}> {}: {};",
                        access_str, binding.name, type_name
                    )
                }
                BindingType::UniformBuffer(type_name) => {
                    format!("var<uniform> {}: {};", binding.name, type_name)
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
            )
            .unwrap();
        }
        if !self.bindings.is_empty() {
            writeln!(shader).unwrap();
        }

        // Add variables
        for variable in &self.variables {
            if let Some(storage_class) = &variable.storage_class {
                if let Some(initial_value) = &variable.initial_value {
                    writeln!(
                        shader,
                        "var<{}> {}: {} = {};",
                        storage_class, variable.name, variable.var_type, initial_value
                    )
                    .unwrap();
                } else {
                    writeln!(
                        shader,
                        "var<{}> {}: {};",
                        storage_class, variable.name, variable.var_type
                    )
                    .unwrap();
                }
            } else {
                if let Some(initial_value) = &variable.initial_value {
                    writeln!(
                        shader,
                        "var {}: {} = {};",
                        variable.name, variable.var_type, initial_value
                    )
                    .unwrap();
                } else {
                    writeln!(shader, "var {}: {};", variable.name, variable.var_type).unwrap();
                }
            }
        }
        if !self.variables.is_empty() {
            writeln!(shader).unwrap();
        }

        // Add functions
        for function in &self.functions {
            let params = function
                .parameters
                .iter()
                .map(|p| format!("{}: {}", p.name, p.param_type))
                .collect::<Vec<_>>()
                .join(", ");

            if let Some(return_type) = &function.return_type {
                writeln!(
                    shader,
                    "fn {}({}) -> {} {{",
                    function.name, params, return_type
                )
                .unwrap();
            } else {
                writeln!(shader, "fn {}({}) {{", function.name, params).unwrap();
            }

            // Add function body with proper indentation
            for line in function.body.lines() {
                if !line.trim().is_empty() {
                    writeln!(shader, "    {}", line).unwrap();
                } else {
                    writeln!(shader).unwrap();
                }
            }
            writeln!(shader, "}}").unwrap();
            writeln!(shader).unwrap();
        }

        // Add main compute function
        let workgroup = if let Some(wg) = &self.workgroup_size {
            match (wg.y, wg.z) {
                (Some(y), Some(z)) => format!("@compute @workgroup_size({}, {}, {})", wg.x, y, z),
                (Some(y), None) => format!("@compute @workgroup_size({}, {})", wg.x, y),
                (None, None) => format!("@compute @workgroup_size({})", wg.x),
                (None, Some(_)) => format!("@compute @workgroup_size({})", wg.x), // Invalid case, ignore z
            }
        } else {
            "@compute @workgroup_size(1)".to_string()
        };

        writeln!(shader, "{}", workgroup).unwrap();
        writeln!(
            shader,
            "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{"
        )
        .unwrap();

        for line in &self.main_body {
            if !line.trim().is_empty() {
                writeln!(shader, "    {}", line).unwrap();
            } else {
                writeln!(shader).unwrap();
            }
        }

        writeln!(shader, "}}").unwrap();

        shader
    }
}

impl Default for ShaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for struct definitions.
pub struct StructBuilder {
    shader_builder: ShaderBuilder,
    struct_name: String,
    fields: Vec<StructField>,
}

impl StructBuilder {
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

    /// Finish building the struct and return to the shader builder.
    pub fn finish_struct(mut self) -> ShaderBuilder {
        self.shader_builder.structs.push(Struct {
            name: self.struct_name,
            fields: self.fields,
        });
        self.shader_builder
    }
}

/// Builder for function definitions.
pub struct FunctionBuilder {
    shader_builder: ShaderBuilder,
    function_name: String,
    parameters: Vec<Parameter>,
    return_type: Option<String>,
    body: String,
}

impl FunctionBuilder {
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

    /// Set the function body.
    pub fn body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// Add a line to the function body.
    pub fn add_body_line(mut self, line: impl Into<String>) -> Self {
        if !self.body.is_empty() {
            self.body.push('\n');
        }
        self.body.push_str(&line.into());
        self
    }

    /// Finish building the function and return to the shader builder.
    pub fn finish_function(mut self) -> ShaderBuilder {
        self.shader_builder.functions.push(Function {
            name: self.function_name,
            parameters: self.parameters,
            return_type: self.return_type,
            body: self.body,
        });
        self.shader_builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple vector addition shader.
    pub fn vector_addition_shader(size: u32) -> String {
        ShaderBuilder::new()
            .workgroup_size(64, None, None)
            .add_storage_buffer(0, 0, "a", "array<f32>", AccessMode::Read)
            .add_storage_buffer(0, 1, "b", "array<f32>", AccessMode::Read)
            .add_storage_buffer(0, 2, "result", "array<f32>", AccessMode::Write)
            .add_main_body_line("let index = global_id.x;")
            .add_main_body_line(format!("if (index >= {}u) {{ return; }}", size))
            .add_main_body_line("result[index] = a[index] + b[index];")
            .build()
    }

    /// Create a matrix multiplication shader.
    pub fn matrix_multiply_shader() -> String {
        ShaderBuilder::new()
            .workgroup_size(16, Some(16), None)
            .add_struct("Matrix")
            .add_field("rows", "u32")
            .add_field("cols", "u32")
            .add_field("data", "array<f32>")
            .finish_struct()
            .add_storage_buffer(0, 0, "matrix_a", "Matrix", AccessMode::Read)
            .add_storage_buffer(0, 1, "matrix_b", "Matrix", AccessMode::Read)
            .add_storage_buffer(0, 2, "result", "Matrix", AccessMode::Write)
            .add_main_body_line("let row = global_id.y;")
            .add_main_body_line("let col = global_id.x;")
            .add_main_body_line("if (row >= matrix_a.rows || col >= matrix_b.cols) { return; }")
            .add_main_body_line("var sum: f32 = 0.0;")
            .add_for_loop(
                "k",
                "0",
                "matrix_a.cols",
                vec![
                    "let a_index = row * matrix_a.cols + k;".to_string(),
                    "let b_index = k * matrix_b.cols + col;".to_string(),
                    "sum += matrix_a.data[a_index] * matrix_b.data[b_index];".to_string(),
                ],
            )
            .add_main_body_line("let result_index = row * matrix_b.cols + col;")
            .add_main_body_line("result.data[result_index] = sum;")
            .build()
    }

    /// Create an image blur shader.
    pub fn image_blur_shader() -> String {
        ShaderBuilder::new()
            .workgroup_size(8, Some(8), None)
            .add_storage_buffer(0, 0, "input_image", "array<vec4<f32>>", AccessMode::Read)
            .add_storage_buffer(0, 1, "output_image", "array<vec4<f32>>", AccessMode::Write)
            .add_uniform_buffer(0, 2, "params", "BlurParams")
            .add_struct("BlurParams")
                .add_field("width", "u32")
                .add_field("height", "u32")
                .add_field("blur_radius", "u32")
                .finish_struct()
            .add_function("get_pixel")
                .add_parameter("x", "i32")
                .add_parameter("y", "i32")
                .return_type("vec4<f32>")
                .body("if (x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.height)) {\n    return vec4<f32>(0.0);\n}\nlet index = u32(y) * params.width + u32(x);\nreturn input_image[index];")
                .finish_function()
            .add_main_body_line("let x = i32(global_id.x);")
            .add_main_body_line("let y = i32(global_id.y);")
            .add_main_body_line("if (x >= i32(params.width) || y >= i32(params.height)) { return; }")
            .add_main_body_line("var color = vec4<f32>(0.0);")
            .add_main_body_line("var weight_sum = 0.0;")
            .add_for_loop(
                "dy",
                format!("-{}", 2),
                "3",
                vec![
                    "for (var dx: i32 = -2; dx <= 2; dx++) {".to_string(),
                    "    let sample_x = x + dx;".to_string(),
                    "    let sample_y = y + dy;".to_string(),
                    "    let weight = 1.0 / (1.0 + f32(dx * dx + dy * dy));".to_string(),
                    "    color += get_pixel(sample_x, sample_y) * weight;".to_string(),
                    "    weight_sum += weight;".to_string(),
                    "}".to_string(),
                ]
            )
            .add_main_body_line("color /= weight_sum;")
            .add_main_body_line("let output_index = u32(y) * params.width + u32(x);")
            .add_main_body_line("output_image[output_index] = color;")
            .build()
    }

    /// Create a particle simulation shader.
    pub fn particle_simulation_shader() -> String {
        ShaderBuilder::new()
            .workgroup_size(256, None, None)
            .add_struct("Particle")
            .add_field("position", "vec3<f32>")
            .add_field("velocity", "vec3<f32>")
            .add_field_with_attributes("mass", "f32", vec!["@align(16)".to_string()])
            .finish_struct()
            .add_struct("SimulationParams")
            .add_field("delta_time", "f32")
            .add_field("gravity", "vec3<f32>")
            .add_field("damping", "f32")
            .add_field("particle_count", "u32")
            .finish_struct()
            .add_storage_buffer(0, 0, "particles", "array<Particle>", AccessMode::ReadWrite)
            .add_uniform_buffer(0, 1, "params", "SimulationParams")
            .add_constant("EPSILON", "f32", "0.001")
            .add_function("apply_gravity")
            .add_parameter("particle", "ptr<storage, Particle, read_write>")
            .body("(*particle).velocity += params.gravity * params.delta_time;")
            .finish_function()
            .add_function("apply_damping")
            .add_parameter("particle", "ptr<storage, Particle, read_write>")
            .body("(*particle).velocity *= params.damping;")
            .finish_function()
            .add_function("update_position")
            .add_parameter("particle", "ptr<storage, Particle, read_write>")
            .body("(*particle).position += (*particle).velocity * params.delta_time;")
            .finish_function()
            .add_main_body_line("let index = global_id.x;")
            .add_main_body_line("if (index >= params.particle_count) { return; }")
            .add_main_body_line("apply_gravity(&particles[index]);")
            .add_main_body_line("apply_damping(&particles[index]);")
            .add_main_body_line("update_position(&particles[index]);")
            .build()
    }

    #[test]
    fn test_vector_addition_shader() {
        let shader = vector_addition_shader(1024);

        // check that the shader contains expected elements
        assert!(shader.contains("@compute @workgroup_size(64)"));
        assert!(shader.contains("@group(0) @binding(0) var<storage, read> a: array<f32>;"));
        assert!(shader.contains("@group(0) @binding(1) var<storage, read> b: array<f32>;"));
        assert!(shader.contains("@group(0) @binding(2) var<storage, write> result: array<f32>;"));
        assert!(shader.contains("let index = global_id.x;"));
        assert!(shader.contains("if (index >= 1024u) { return; }"));
        assert!(shader.contains("result[index] = a[index] + b[index];"));
    }

    #[test]
    fn test_matrix_multiply_shader() {
        let shader = matrix_multiply_shader();

        // check workgroup size
        assert!(shader.contains("@compute @workgroup_size(16, 16)"));

        // check struct definition
        assert!(shader.contains("struct Matrix {"));
        assert!(shader.contains("rows: u32,"));
        assert!(shader.contains("cols: u32,"));
        assert!(shader.contains("data: array<f32>,"));

        // check bindings
        assert!(shader.contains("@group(0) @binding(0) var<storage, read> matrix_a: Matrix;"));
        assert!(shader.contains("@group(0) @binding(1) var<storage, read> matrix_b: Matrix;"));
        assert!(shader.contains("@group(0) @binding(2) var<storage, write> result: Matrix;"));

        // check main function logic
        assert!(shader.contains("let row = global_id.y;"));
        assert!(shader.contains("let col = global_id.x;"));
        assert!(shader.contains("var sum: f32 = 0.0;"));

        // check for loop
        assert!(shader.contains("for (var k: u32 = 0u; k < matrix_a.cols; k++) {"));
    }

    #[test]
    fn test_image_blur_shader() {
        let shader = image_blur_shader();

        // check workgroup size
        assert!(shader.contains("@compute @workgroup_size(8, 8)"));

        // check struct
        assert!(shader.contains("struct BlurParams {"));
        assert!(shader.contains("width: u32,"));
        assert!(shader.contains("height: u32,"));
        assert!(shader.contains("blur_radius: u32,"));

        // check bindings
        assert!(
            shader.contains(
                "@group(0) @binding(0) var<storage, read> input_image: array<vec4<f32>>;"
            )
        );
        assert!(
            shader.contains(
                "@group(0) @binding(1) var<storage, write> output_image: array<vec4<f32>>;"
            )
        );
        assert!(shader.contains("@group(0) @binding(2) var<uniform> params: BlurParams;"));

        // check function
        assert!(shader.contains("fn get_pixel(x: i32, y: i32) -> vec4<f32> {"));

        // check main logic
        assert!(shader.contains("let x = i32(global_id.x);"));
        assert!(shader.contains("let y = i32(global_id.y);"));
        assert!(shader.contains("var color = vec4<f32>(0.0);"));
    }

    #[test]
    fn test_particle_simulation_shader() {
        let shader = particle_simulation_shader();

        // check workgroup size
        assert!(shader.contains("@compute @workgroup_size(256)"));

        // check structs
        assert!(shader.contains("struct Particle {"));
        assert!(shader.contains("position: vec3<f32>,"));
        assert!(shader.contains("velocity: vec3<f32>,"));
        assert!(shader.contains("@align(16) mass: f32,"));

        assert!(shader.contains("struct SimulationParams {"));
        assert!(shader.contains("delta_time: f32,"));
        assert!(shader.contains("gravity: vec3<f32>,"));

        // check constant
        assert!(shader.contains("const EPSILON: f32 = 0.001;"));

        // check functions
        assert!(
            shader.contains("fn apply_gravity(particle: ptr<storage, Particle, read_write>) {")
        );
        assert!(
            shader.contains("fn apply_damping(particle: ptr<storage, Particle, read_write>) {")
        );
        assert!(
            shader.contains("fn update_position(particle: ptr<storage, Particle, read_write>) {")
        );

        // check main logic
        assert!(shader.contains("let index = global_id.x;"));
        assert!(shader.contains("if (index >= params.particle_count) { return; }"));
        assert!(shader.contains("apply_gravity(&particles[index]);"));
        assert!(shader.contains("apply_damping(&particles[index]);"));
        assert!(shader.contains("update_position(&particles[index]);"));
    }

    #[test]
    fn test_builder_pattern() {
        let shader = ShaderBuilder::new()
            .workgroup_size(32, None, None)
            .add_constant("PI", "f32", "3.14159")
            .add_storage_buffer(0, 0, "data", "array<f32>", AccessMode::ReadWrite)
            .add_main_body_line("let index = global_id.x;")
            .add_main_body_line("data[index] *= PI;")
            .build();

        assert!(shader.contains("const PI: f32 = 3.14159;"));
        assert!(shader.contains("@compute @workgroup_size(32)"));
        assert!(
            shader.contains("@group(0) @binding(0) var<storage, read_write> data: array<f32>;")
        );
        assert!(shader.contains("let index = global_id.x;"));
        assert!(shader.contains("data[index] *= PI;"));
    }
}
