use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::anyhow;
use cow_utils::CowUtils;
use vello::wgpu::{
    Adapter, Backends, CompositeAlphaMode, Instance, PowerPreference, Queue, Surface, SurfaceConfiguration, Texture, TextureView,
};
use vello::wgpu::{Device, TextureFormat};
use vello::{AaSupport, Renderer as VelloRenderer, RendererOptions as VelloRendererOptions};

use gosub_interface::render_backend::WindowHandle;
use gosub_shared::types::Result;

pub mod window;

pub const RENDERER_CONF: VelloRendererOptions = VelloRendererOptions {
    use_cpu: false,
    antialiasing_support: AaSupport {
        area: true,
        msaa8: true,
        msaa16: true,
    },
    num_init_threads: NonZeroUsize::new(1),
    pipeline_cache: None,
};

#[derive(Clone, Debug)]
pub struct Renderer {
    pub instance_adapter: Arc<InstanceAdapter>,
}

#[derive(Debug)]
pub struct InstanceAdapter {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

pub struct RendererOptions {
    pub power_preference: Option<PowerPreference>,
    #[cfg(not(target_arch = "wasm32"))]
    pub adapter: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    pub crash_on_invalid_adapter: bool,
}

impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            power_preference: PowerPreference::from_env(),
            #[cfg(not(target_arch = "wasm32"))]
            adapter: std::env::var("WGPU_ADAPTER_NAME").ok(),
            #[cfg(not(target_arch = "wasm32"))]
            crash_on_invalid_adapter: false,
        }
    }
}

struct RenderConfig {
    pub power_preference: PowerPreference,
    #[cfg(not(target_arch = "wasm32"))]
    pub adapter: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    pub crash_on_invalid_adapter: bool,
}

impl From<RendererOptions> for RenderConfig {
    fn from(opts: RendererOptions) -> Self {
        Self {
            power_preference: opts
                .power_preference
                .unwrap_or(PowerPreference::from_env().unwrap_or_default()),
            #[cfg(not(target_arch = "wasm32"))]
            adapter: opts.adapter,
            #[cfg(not(target_arch = "wasm32"))]
            crash_on_invalid_adapter: opts.crash_on_invalid_adapter,
        }
    }
}

impl Renderer {
    pub async fn new(opts: RendererOptions) -> Result<Self> {
        let config = RenderConfig::from(opts);

        Ok(Self {
            instance_adapter: Arc::new(Self::get_adapter(config).await?),
        })
    }

    async fn get_adapter(config: RenderConfig) -> Result<InstanceAdapter> {
        let instance = Instance::new(&vello::wgpu::InstanceDescriptor {
            backends: vello::wgpu::Backends::from_env().unwrap_or_default(),
            flags: vello::wgpu::InstanceFlags::from_build_config().with_env(),
            backend_options: vello::wgpu::BackendOptions::from_env_or_default(),
        });

        #[cfg(not(target_arch = "wasm32"))]
        let mut adapter = config.adapter.and_then(|adapter_name| {
            let adapters = instance.enumerate_adapters(Backends::all());
            let adapter_name = adapter_name.cow_to_lowercase();

            let mut chosen_adapter = None;
            for adapter in adapters {
                let info = adapter.get_info();

                if info.name.cow_to_lowercase().contains(adapter_name.as_ref()) {
                    chosen_adapter = Some(adapter);
                    break;
                }
            }

            if chosen_adapter.is_none() && config.crash_on_invalid_adapter {
                eprintln!("No adapter found with name: {}", adapter_name);
                std::process::exit(1);
            }

            chosen_adapter
        });

        #[cfg(target_arch = "wasm32")]
        let mut adapter = None;

        if adapter.is_none() {
            adapter = instance
                .request_adapter(&vello::wgpu::RequestAdapterOptions {
                    power_preference: config.power_preference,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await;
        }

        if adapter.is_none() {
            adapter = instance
                .request_adapter(&vello::wgpu::RequestAdapterOptions {
                    power_preference: config.power_preference,
                    force_fallback_adapter: true,
                    compatible_surface: None,
                })
                .await;
        }

        let adapter = adapter.ok_or(anyhow!("No adapter found"))?;

        let info = adapter.get_info();

        let mut features = adapter.features();

        if info.device_type == vello::wgpu::DeviceType::DiscreteGpu {
            features -= vello::wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }

        // features -= vello::wgpu::Features::RAY_QUERY;
        // features -= vello::wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE;

        let (device, queue) = adapter
            .request_device(
                &vello::wgpu::DeviceDescriptor {
                    label: None,
                    required_features: Default::default(),
                    required_limits: Default::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| anyhow!(e.to_string()))?;

        Ok(InstanceAdapter {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

pub struct SurfaceWrapper<'a> {
    pub surface: Surface<'a>,
    pub config: SurfaceConfiguration,
    
    pub target_texture: Texture,
    pub target_view: TextureView,
}

impl InstanceAdapter {
    pub fn create_renderer(&self) -> Result<VelloRenderer> {
        VelloRenderer::new(&self.device, RENDERER_CONF).map_err(|e| anyhow!(e.to_string()))
    }
    pub fn create_surface<'a>(
        &self,
        window: impl WindowHandle + 'a,
        width: u32,
        height: u32,
        present_mode: vello::wgpu::PresentMode,
    ) -> Result<SurfaceWrapper<'a>> {
        let surface = self.instance.create_surface(window)?;
        let capabilities = surface.get_capabilities(&self.adapter);
        let format = capabilities
            .formats
            .into_iter()
            .find(|it| matches!(it, TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm))
            .ok_or(anyhow!("surface should support Rgba8Unorm or Bgra8Unorm"))?;

        let config = SurfaceConfiguration {
            usage: vello::wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        
        let (target_texture, target_view) =
            create_intermediate_texture(width, height, &self.device);

        let surface = SurfaceWrapper { surface, config, target_texture, target_view };

        self.configure_surface(&surface);

        Ok(surface)
    }

    pub fn resize_surface(&self, surface: &mut SurfaceWrapper, width: u32, height: u32) {
        surface.config.width = width;
        surface.config.height = height;
        
        let (target_texture, target_view) =
            create_intermediate_texture(width, height, &self.device);
        
        surface.target_texture = target_texture;
        surface.target_view = target_view;
        
        self.configure_surface(surface);
    }

    fn configure_surface(&self, surface: &SurfaceWrapper) {
        surface.surface.configure(&self.device, &surface.config);
    }
}

fn create_intermediate_texture(width: u32, height: u32, device: &Device) -> (Texture, TextureView) {
    let target_texture = device.create_texture(&vello::wgpu::TextureDescriptor {
        label: None,
        size: vello::wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: vello::wgpu::TextureDimension::D2,
        usage: vello::wgpu::TextureUsages::STORAGE_BINDING | vello::wgpu::TextureUsages::TEXTURE_BINDING,
        format: TextureFormat::Rgba8Unorm,
        view_formats: &[],
    });
    let target_view = target_texture.create_view(&vello::wgpu::TextureViewDescriptor::default());
    (target_texture, target_view)
}
