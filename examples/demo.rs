use cgmath::*;
use eframe::egui;
use newell::NewellObj;
use serde::{Deserialize, Serialize};

fn main() {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Newell's Algorithm Demo",
        options,
        Box::new(|_cc| Box::new(NewellDemo::default())),
    );
}

#[derive(Debug, Default)]
struct NewellDemo;
impl eframe::App for NewellDemo {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let pitch_id = egui::Id::new("pitch");
        let yaw_id = egui::Id::new("yaw");
        let splits_id = egui::Id::new("splits");
        let polygons_id = egui::Id::new("polygons");

        let mut polygons = ctx
            .data()
            .get_persisted_mut_or_default::<Option<Vec<Polygon>>>(polygons_id)
            .take()
            .unwrap_or_default();

        let mut newell_polys: Vec<_> = polygons
            .iter()
            .filter_map(Polygon::to_newell_polygon)
            .collect();
        let max_splits: usize = ctx.data().get_persisted(splits_id).unwrap_or(0);
        let mut log = String::new();
        let splits_done = newell::sort(&mut newell_polys, max_splits, &mut log);

        egui::SidePanel::new(egui::containers::panel::Side::Left, "left").show(ctx, |ui| {
            let next_id = polygons.iter().map(|p| p.id).max().unwrap_or(0) + 1;

            egui::ScrollArea::new([false, true]).show(ui, |ui| {
                let mut reorder = None;
                let mut i: i32 = -1;
                let poly_count = polygons.len();
                polygons.retain_mut(|p| {
                    i += 1;

                    egui::CollapsingHeader::new("Polygon")
                        .id_source(p.id)
                        .default_open(true)
                        .show(ui, |ui| {
                            let mut delete = false;
                            ui.horizontal(|ui| {
                                delete = ui.button("Delete").clicked();
                                if ui.button("^").clicked() && i > 0 {
                                    reorder = Some((i, i - 1));
                                }
                                if ui.button("v").clicked() && i + 1 < poly_count as i32 {
                                    reorder = Some((i, i + 1));
                                }
                            });
                            ui.text_edit_singleline(&mut p.name);

                            ui.label("Offset");
                            ui.horizontal(|ui| {
                                ui.add(egui::DragValue::new(&mut p.offset.x).speed(0.1));
                                ui.add(egui::DragValue::new(&mut p.offset.y).speed(0.1));
                                ui.add(egui::DragValue::new(&mut p.offset.z).speed(0.1));
                            });

                            ui.label("Rotation");
                            ui.horizontal(|ui| {
                                ui.drag_angle(&mut p.theta);
                                p.theta = p.theta.rem_euclid(std::f32::consts::TAU);
                                ui.drag_angle(&mut p.phi);
                                p.phi = p.phi.clamp(
                                    -std::f32::consts::FRAC_PI_2,
                                    std::f32::consts::FRAC_PI_2,
                                )
                            });

                            ui.label("Points");
                            p.verts.retain_mut(|p| {
                                ui.horizontal(|ui| {
                                    let delete = ui.button("Delete").clicked();
                                    ui.add(egui::DragValue::new(&mut p.x).speed(0.01));
                                    ui.add(egui::DragValue::new(&mut p.y).speed(0.01));
                                    !delete
                                })
                                .inner
                            });
                            if ui.button("Add").clicked() {
                                p.verts.push(vec2(0.0, 0.0));
                            }

                            !delete
                        })
                        .body_returned
                        .unwrap_or(true)
                });

                if let Some((from, to)) = reorder {
                    polygons.swap(from as usize, to as usize);
                }
            });

            if ui.button("Add").clicked() {
                polygons.push(Polygon {
                    id: next_id,
                    name: (('A' as u8 + next_id as u8) as char).to_string(),
                    ..Default::default()
                });
            }
        });

        egui::SidePanel::new(egui::containers::panel::Side::Right, "right").show(ctx, |ui| {
            ui.label("Pitch");
            let mut pitch = ui.data().get_persisted(pitch_id).unwrap_or(0.0);
            ui.drag_angle(&mut pitch);
            ui.data().insert_persisted(
                pitch_id,
                pitch.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2),
            );

            ui.label("Yaw");
            let mut yaw = ui.data().get_persisted(yaw_id).unwrap_or(0.0);
            ui.drag_angle(&mut yaw);
            ui.data()
                .insert_persisted(yaw_id, yaw.rem_euclid(std::f32::consts::TAU));

            let mut splits: usize = ui.data().get_persisted(splits_id).unwrap_or(0);

            ui.separator();
            ui.label("Max splits");
            ui.add(egui::DragValue::new(&mut splits).speed(0.1));
            ui.data().insert_persisted(splits_id, splits);
            ui.label(format!("Used {splits_done} splits"));

            ui.separator();
            ui.label("Final order:");
            let mut order = String::new();
            for p in &newell_polys {
                order += p.name();
                order += "\n";
            }
            ui.add(egui::TextEdit::multiline(&mut order).interactive(false));

            ui.separator();
            ui.label("Log:");
            egui::ScrollArea::new([false, true]).show(ui, |ui| {
                ui.add(egui::TextEdit::multiline(&mut log).interactive(false));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Newell's Algorithm Demo");

            let pitch: f32 = ui.data().get_persisted(pitch_id).unwrap_or(0.0);
            let yaw: f32 = ui.data().get_persisted(yaw_id).unwrap_or(0.0);
            egui::plot::Plot::new("polygon_plot")
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    let rot = Matrix3::from_angle_x(Rad(pitch)) * Matrix3::from_angle_y(Rad(yaw));
                    for p in &newell_polys {
                        plot_ui.polygon(
                            egui::plot::Polygon::new(egui::plot::Values::from_values_iter(
                                p.verts()
                                    .iter()
                                    .map(|&p| rot.transform_point(p))
                                    .map(|xy| egui::plot::Value::new(xy.x, xy.y)),
                            ))
                            .name(p.name()),
                        );
                    }
                });
        });

        *ctx.data().get_persisted_mut_or_default(polygons_id) = Some(polygons);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Polygon {
    id: u64,
    #[serde(default)]
    name: String,

    offset: Vector3<f32>,

    theta: f32,
    phi: f32,

    verts: Vec<Vector2<f32>>,
}
impl Default for Polygon {
    fn default() -> Self {
        Self {
            id: 0,
            name: "A".to_owned(),

            offset: Vector3::zero(),

            theta: 0.0,
            phi: 0.0,

            verts: vec![
                Vector2::new(-1.0, -1.0),
                Vector2::new(1.0, -1.0),
                Vector2::new(1.0, 1.0),
                Vector2::new(-1.0, 1.0),
            ],
        }
    }
}
impl Polygon {
    fn to_newell_polygon(&self) -> Option<newell::Polygon> {
        let rot = Matrix3::from_angle_z(Rad(self.theta)) * Matrix3::from_angle_x(Rad(self.phi));
        newell::Polygon::new(
            self.verts
                .iter()
                .map(|v| rot * v.extend(0.0) + self.offset)
                .map(Point3::from_vec)
                .collect(),
            self.name.clone(),
        )
    }
}
