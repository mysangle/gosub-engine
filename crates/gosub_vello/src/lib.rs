use std::fmt::Debug;

use anyhow::anyhow;
use gosub_fontmanager::FontManager;
use log::info;
use vello::kurbo::Point as VelloPoint;
use vello::peniko::Color as VelloColor;
use vello::wgpu::{CommandEncoderDescriptor, Device, Texture, TextureFormat, TextureView, TextureViewDescriptor, util::TextureBlitter};
use vello::{AaConfig, RenderParams, Scene as VelloScene};

pub use border::*;
pub use brush::*;
pub use color::*;
use gosub_interface::font::HasFontManager;
use gosub_interface::render_backend::{RenderBackend, RenderRect, RenderText, Scene as TScene, WindowHandle};
use gosub_shared::geo::{Point, SizeU32};
use gosub_shared::types::Result;
pub use gradient::*;
pub use image::*;
pub use rect::*;
pub use scene::*;
pub use text::*;
pub use transform::*;

use crate::render::window::{ActiveWindowData, WindowData};
use crate::render::{Renderer, RendererOptions};

mod border;
mod brush;
mod color;
mod gradient;
mod image;
mod rect;
mod render;
mod scene;
mod text;
mod transform;

mod debug;
#[cfg(feature = "vello_svg")]
mod vello_svg;

pub struct VelloBackend {
    #[cfg(target_arch = "wasm32")]
    renderer: Renderer,
}

impl Debug for VelloBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VelloRenderer").finish()
    }
}

impl HasFontManager for VelloBackend {
    type FontManager = FontManager;
}

impl RenderBackend for VelloBackend {
    type Rect = Rect;
    type Border = Border;
    type BorderSide = BorderSide;
    type BorderRadius = BorderRadius;
    type Transform = Transform;
    type Gradient = Gradient;
    type Color = Color;
    type Image = Image;
    type Brush = Brush;
    type Scene = Scene;
    type Text = Text;
    #[cfg(feature = "resvg")]
    type SVGRenderer = gosub_svg::resvg::Resvg;
    #[cfg(all(feature = "vello_svg", not(feature = "resvg")))]
    type SVGRenderer = vello_svg::VelloSVG;

    type ActiveWindowData<'a> = ActiveWindowData<'a>;
    type WindowData<'a> = WindowData;

    type FontManager = FontManager;

    fn draw_rect(&mut self, data: &mut Self::WindowData<'_>, rect: &RenderRect<Self>) {
        data.scene.draw_rect(rect);
    }

    fn draw_text(&mut self, data: &mut Self::WindowData<'_>, text: &RenderText<Self>) {
        data.scene.draw_text(text);
    }

    fn apply_scene(
        &mut self,
        data: &mut Self::WindowData<'_>,
        scene: &Self::Scene,
        transform: Option<Self::Transform>,
    ) {
        data.scene.apply_scene(scene, transform);
    }

    fn reset(&mut self, data: &mut Self::WindowData<'_>) {
        data.scene.reset();
    }

    fn activate_window<'a>(
        &mut self,
        handle: impl WindowHandle + 'a,
        data: &mut Self::WindowData<'_>,
        size: SizeU32,
    ) -> Result<Self::ActiveWindowData<'a>> {
        let surface =
            data.adapter
                .create_surface(handle, size.width, size.height, vello::wgpu::PresentMode::AutoVsync)?;

        let renderer = data.adapter.create_renderer()?;

        data.renderer = renderer;

        Ok(ActiveWindowData { surface })
    }

    fn suspend_window(
        &mut self,
        _handle: impl WindowHandle,
        _data: &mut Self::ActiveWindowData<'_>,
        _window_data: &mut Self::WindowData<'_>,
    ) -> Result<()> {
        Ok(())
    }

    fn create_window_data<'a>(&mut self, _handle: impl WindowHandle) -> Result<Self::WindowData<'a>> {
        info!("Creating window data");

        #[cfg(target_arch = "wasm32")]
        let renderer = self.renderer.clone();

        #[cfg(not(target_arch = "wasm32"))]
        let renderer = futures::executor::block_on(Renderer::new(RendererOptions::default()))?;

        let adapter = renderer.instance_adapter;

        let renderer = adapter.create_renderer()?;

        info!("Created renderer");

        Ok(WindowData {
            adapter,
            renderer,
            scene: VelloScene::new().into(),
        })
    }

    fn resize_window<'a>(
        &mut self,
        window_data: &mut Self::WindowData<'a>,
        active_window_data: &mut Self::ActiveWindowData<'a>,
        size: SizeU32,
    ) -> Result<()> {
        window_data
            .adapter
            .resize_surface(&mut active_window_data.surface, size.width, size.height);

        Ok(())
    }

    fn render<'a>(
        &mut self,
        window_data: &mut Self::WindowData<'a>,
        active_data: &mut Self::ActiveWindowData<'a>,
    ) -> Result<()> {
        let height = active_data.surface.config.height;
        let width = active_data.surface.config.width;

        let surface_texture = active_data.surface.surface.get_current_texture()?;
        let (target_texture, target_view) =
            create_intermediate_texture(width, height, &window_data.adapter.device);

        window_data
            .renderer
            .render_to_texture(
                &window_data.adapter.device,
                &window_data.adapter.queue,
                &window_data.scene.0,
                &target_view,
                &RenderParams {
                    base_color: VelloColor::WHITE,
                    width,
                    height,
                    antialiasing_method: AaConfig::Msaa16,
                },
            )
            .map_err(|e| anyhow!(e.to_string()))?;
        
        let mut encoder = window_data.adapter
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Surface Blit"),
            });

        let blitter = TextureBlitter::new(&window_data.adapter.device, active_data.surface.config.format);
        blitter.copy(
            &window_data.adapter.device,
            &mut encoder,
            &target_view,
            &surface_texture
                .texture
                .create_view(&TextureViewDescriptor::default()),
        );
        window_data.adapter.queue.submit([encoder.finish()]);

        surface_texture.present();
        
        window_data.adapter.device.poll(vello::wgpu::Maintain::Wait);

        Ok(())
    }
}

impl VelloBackend {
    #[cfg(target_arch = "wasm32")]
    pub async fn new() -> Result<Self> {
        let renderer = Renderer::new(RendererOptions::default()).await?;

        Ok(Self { renderer })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for VelloBackend {
    fn default() -> Self {
        Self::new()
    }
}

trait Convert<T> {
    fn convert(self) -> T;
}

impl Convert<VelloPoint> for Point {
    fn convert(self) -> VelloPoint {
        VelloPoint::new(self.x as f64, self.y as f64)
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
