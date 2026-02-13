import NiiViewer from "./NiiViewer";

export default function ImagingPanel() {
  return (
    <section className="card imaging">
      <div className="card-header">
        <h2>Imaging Review</h2>
        <p>T2W + ADC volumes with slice navigation.</p>
      </div>
      <div className="image-grid">
        <NiiViewer title="T2W" subtitle="Axial T2-weighted" url="/imgs/t2w.nii.gz" />
        <NiiViewer title="ADC" subtitle="Diffusion-weighted" url="/imgs/adc.nii.gz" />
      </div>
    </section>
  );
}
