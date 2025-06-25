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
