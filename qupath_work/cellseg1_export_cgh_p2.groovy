import qupath.lib.regions.RegionRequest
import qupath.lib.roi.GeometryTools
import org.locationtech.jts.geom.Coordinate
import javax.imageio.ImageIO
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.RenderingHints
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage

File outDir = new File('/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2')
File trainImages = new File(outDir, 'train/images')
File trainMasks = new File(outDir, 'train/masks')
File auxMasks = new File(outDir, 'auxiliary_masks')
File semanticMasks = new File(outDir, 'semantic_masks')
File previews = new File(outDir, 'previews')
[trainImages, trainMasks, auxMasks, semanticMasks, previews].each { dir ->
    dir.mkdirs()
    dir.listFiles()?.findAll { it.isFile() && (it.getName().endsWith('.png') || it.getName().endsWith('.csv')) }?.each { it.delete() }
}

File instanceCsv = new File(outDir, 'cell_instances.csv')
File tileCsv = new File(outDir, 'dataset_manifest.csv')
File qcCsv = new File(outDir, 'boundary_qc.csv')
instanceCsv.text = 'tile_id,tile_name,instance_label,boundary_name,boundary_class,nucleus_name,boundary_area_px,local_centroid_x,local_centroid_y,wsi_centroid_x,wsi_centroid_y,nucleus_wsi_x,nucleus_wsi_y,stroma_overlap_px,stroma_overlap_fraction\n'
tileCsv.text = 'tile_id,tile_name,image_file,mask_file,width,height,trainable_instances,clear_trainable,compact_trainable,edge_or_invalid_boundary_ignore,uncertain_ignore_regions,nuclei_total,nuclei_in_tile,stroma_regions_intersecting_tile\n'
qcCsv.text = 'tile_id,tile_name,boundary_name,boundary_class,include_for_cellseg1,nuclei_inside,nuclei_inside_in_tile,nuclei_names,stroma_overlap_px,stroma_overlap_fraction,flags\n'

String csvEscape(value) {
    String s = value == null ? '' : value.toString()
    return '"' + s.replace('"', '""') + '"'
}
String csvRow(List values) { return values.collect { csvEscape(it) }.join(',') + '\n' }

def imageData = getCurrentImageData()
def server = imageData.getServer()
def objs = getAnnotationObjects()
def clsOf = { obj -> obj.getPathClass() == null ? '' : obj.getPathClass().toString() }
def gf = GeometryTools.getDefaultFactory()
def pointFor = { obj -> def r = obj.getROI(); gf.createPoint(new Coordinate(r.getCentroidX(), r.getCentroidY())) }
def geomOf = { obj -> GeometryTools.ensurePolygonal(GeometryTools.roiToGeometry(obj.getROI())) }
def positiveClasses = ['GT Clear cell boundary', 'GT Compact cell boundary'] as Set
def uncertainClasses = ['GT Uncertain', 'GT Uncertain clear cell boundary'] as Set

def sortByPosition = { list ->
    list.sort { a, b ->
        int yc = a.getROI().getCentroidY() <=> b.getROI().getCentroidY()
        yc != 0 ? yc : (a.getROI().getCentroidX() <=> b.getROI().getCentroidX())
    }
}

def colorFor = { obj ->
    switch (clsOf(obj)) {
        case 'GT Nucleus': return new Color(0, 90, 255)
        case 'GT Clear cell boundary': return new Color(0, 210, 80)
        case 'GT Compact cell boundary': return new Color(220, 40, 220)
        case 'GT Uncertain clear cell boundary': return new Color(255, 165, 0)
        case 'GT Stroma': return new Color(231, 76, 60)
        default: return Color.WHITE
    }
}

String tileIdForName(String name) {
    def p2 = (name =~ /^P2 tile\s+(\d+)$/)
    if (p2.matches())
        return String.format(Locale.US, 'p2_tile_%02d', Integer.parseInt(p2[0][1]))
    def yolo = (name =~ /^yolo_tile_(\d+)$/)
    if (yolo.matches())
        return String.format(Locale.US, 'yolo_tile_%02d', Integer.parseInt(yolo[0][1]))
    return null
}

int tileSortKey(String name) {
    def p2 = (name =~ /^P2 tile\s+(\d+)$/)
    if (p2.matches())
        return Integer.parseInt(p2[0][1])
    def yolo = (name =~ /^yolo_tile_(\d+)$/)
    if (yolo.matches())
        return Integer.parseInt(yolo[0][1])
    return 100000
}

def tiles = objs.findAll {
    clsOf(it) == 'GT Training tile' && it.getROI() != null && tileIdForName(it.getName() ?: '') != null
}.sort { a, b ->
    int ak = tileSortKey(a.getName() ?: '')
    int bk = tileSortKey(b.getName() ?: '')
    ak != bk ? ak <=> bk : ((a.getName() ?: '') <=> (b.getName() ?: ''))
}.collect {
    [name: it.getName(), id: tileIdForName(it.getName())]
}

if (tiles.isEmpty())
    throw new IllegalArgumentException('No final training tiles found. Expected names like "P2 tile 01" or "yolo_tile_21".')

def summaryRows = []
int totalTrainable = 0
int totalClear = 0
int totalCompact = 0
int totalEdgeIgnore = 0
int totalUncertain = 0
int totalNucleiInTile = 0

tiles.each { spec ->
    def tile = objs.find { (it.getName() ?: '') == spec.name && it.getROI() != null }
    if (tile == null) throw new IllegalArgumentException('Missing training tile: ' + spec.name)
    def tileRoi = tile.getROI()
    def tileGeom = geomOf(tile)
    def centroidInTile = { obj -> obj.getROI() != null && tileGeom.covers(pointFor(obj)) }
    def intersectsTile = { obj ->
        if (obj.getROI() == null) return false
        def g = geomOf(obj)
        return g.intersects(tileGeom) && GeometryTools.ensurePolygonal(g.intersection(tileGeom)).getArea() > 0.0d
    }

    int x = Math.round((float)tileRoi.getBoundsX())
    int y = Math.round((float)tileRoi.getBoundsY())
    int w = Math.round((float)tileRoi.getBoundsWidth())
    int h = Math.round((float)tileRoi.getBoundsHeight())
    if (w <= 0 || h <= 0) throw new IllegalArgumentException('Invalid tile size for ' + spec.name)

    def raw = server.readRegion(RegionRequest.createInstance(server.getPath(), 1.0, x, y, w, h))
    File imageFile = new File(trainImages, spec.id + '.png')
    ImageIO.write(raw, 'PNG', imageFile)

    def tx = AffineTransform.getTranslateInstance(-x, -y)
    def localShape = { geom -> tx.createTransformedShape(GeometryTools.geometryToShape(geom)) }
    def clipGeom = { obj -> GeometryTools.ensurePolygonal(geomOf(obj).intersection(tileGeom)) }

    def drawMask = { File file, List annos, boolean instanceLabels ->
        def mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        def g2 = mask.createGraphics()
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
        int label = 1
        annos.each { obj ->
            def geom = clipGeom(obj)
            if (!geom.isEmpty()) {
                int v = instanceLabels ? label : 255
                v = Math.max(0, Math.min(255, v))
                g2.setColor(new Color(v, v, v))
                g2.fill(localShape(geom))
            }
            label++
        }
        g2.dispose()
        ImageIO.write(mask, 'PNG', file)
    }
    def drawMaskSubtract = { File file, List annos, boolean instanceLabels, subtractGeom ->
        def mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        def g2 = mask.createGraphics()
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
        int label = 1
        annos.each { obj ->
            def geom = clipGeom(obj)
            if (subtractGeom != null && !subtractGeom.isEmpty() && !geom.isEmpty())
                geom = GeometryTools.ensurePolygonal(geom.difference(subtractGeom))
            if (!geom.isEmpty()) {
                int v = instanceLabels ? label : 255
                v = Math.max(0, Math.min(255, v))
                g2.setColor(new Color(v, v, v))
                g2.fill(localShape(geom))
            }
            label++
        }
        g2.dispose()
        ImageIO.write(mask, 'PNG', file)
    }
    def drawSemantic = { File file, Map classValues ->
        def mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        def g2 = mask.createGraphics()
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
        classValues.each { cls, row ->
            int v = row.value as int
            row.objects.each { obj ->
                def geom = clipGeom(obj)
                if (!geom.isEmpty()) {
                    g2.setColor(new Color(v, v, v))
                    g2.fill(localShape(geom))
                }
            }
        }
        g2.dispose()
        ImageIO.write(mask, 'PNG', file)
    }

    def allNuclei = sortByPosition(objs.findAll { clsOf(it) == 'GT Nucleus' && it.getROI() != null })
    def nuclei = sortByPosition(allNuclei.findAll { centroidInTile(it) || intersectsTile(it) })
    def stromas = sortByPosition(objs.findAll { clsOf(it) == 'GT Stroma' && it.getROI() != null && intersectsTile(it) })
    def uncertain = sortByPosition(objs.findAll { uncertainClasses.contains(clsOf(it)) && it.getROI() != null && intersectsTile(it) })
    def boundaries = sortByPosition(objs.findAll { positiveClasses.contains(clsOf(it)) && it.getROI() != null && centroidInTile(it) })
    def nucleusInsideStroma = { n -> stromas.any { s -> geomOf(s).covers(pointFor(n)) } }
    def metadataExcludedForExport = { obj -> (obj.getMetadata().get('exclude_from_training_export') ?: '').toString().equalsIgnoreCase('true') }
    def nucleiInTile = nuclei.findAll { centroidInTile(it) && !metadataExcludedForExport(it) && !nucleusInsideStroma(it) }
    def nucleiAllForExport = nuclei.findAll { !metadataExcludedForExport(it) && !nucleusInsideStroma(it) }

    def validRows = []
    def validBoundaries = []
    def edgeOrInvalid = []
    boundaries.each { b ->
        def bg = geomOf(b)
        def contained = allNuclei.findAll { n -> bg.covers(pointFor(n)) }
        def containedInTile = contained.findAll { centroidInTile(it) }
        def containedInStroma = contained.findAll { nucleusInsideStroma(it) }
        double stromaOverlap = 0d
        stromas.each { s ->
            def sg = geomOf(s)
            if (bg.intersects(sg)) stromaOverlap += GeometryTools.ensurePolygonal(bg.intersection(sg)).getArea()
        }
        double area = GeometryTools.ensurePolygonal(bg.intersection(tileGeom)).getArea()
        double frac = area > 0 ? stromaOverlap / area : 0d
        def flags = []
        if (contained.size() == 0) flags << 'ZERO_NUCLEUS'
        if (contained.size() > 1) flags << 'MULTI_NUCLEI'
        if (contained.size() == 1 && containedInTile.size() == 0) flags << 'NUCLEUS_OUTSIDE_TILE'
        if (!containedInStroma.isEmpty()) flags << 'NUCLEUS_INSIDE_STROMA'
        if (frac > 0.02) flags << 'STROMA_OVERLAP'
        boolean metadataExcluded = (b.getMetadata().get('exclude_from_training_export') ?: '').toString().equalsIgnoreCase('true')
        boolean include = !metadataExcluded && containedInTile.size() == 1 && contained.size() == 1 && containedInStroma.isEmpty()
        if (include) {
            validBoundaries << b
            validRows << [boundary: b, nucleus: containedInTile[0], area: area, overlap: stromaOverlap, frac: frac, flags: flags]
        } else {
            edgeOrInvalid << b
        }
        qcCsv << csvRow([spec.id, spec.name, b.getName(), clsOf(b), include, contained.size(), containedInTile.size(), contained.collect { it.getName() ?: '' }.join('|'), String.format('%.1f', stromaOverlap), String.format('%.4f', frac), flags.isEmpty() ? 'OK' : flags.join('|')])
    }
    def positiveSubtractGeom = validBoundaries.isEmpty() ? null :
        GeometryTools.ensurePolygonal(GeometryTools.union(validBoundaries.collect { geomOf(it) }).buffer(0.5d).intersection(tileGeom))

    File maskFile = new File(trainMasks, spec.id + '.png')
    drawMask(maskFile, validBoundaries, true)
    drawMask(new File(auxMasks, spec.id + '_gt_nucleus_instances.png'), nucleiInTile, true)
    drawMask(new File(auxMasks, spec.id + '_gt_nucleus_all_direct_children.png'), nucleiAllForExport, true)
    drawMaskSubtract(new File(auxMasks, spec.id + '_gt_stroma.png'), stromas, false, positiveSubtractGeom)
    drawMaskSubtract(new File(auxMasks, spec.id + '_gt_uncertain_ignore.png'), uncertain, false, positiveSubtractGeom)
    drawMaskSubtract(new File(auxMasks, spec.id + '_edge_or_invalid_cell_ignore.png'), edgeOrInvalid, false, positiveSubtractGeom)
    drawMask(new File(auxMasks, spec.id + '_gt_clear_boundary_all.png'), boundaries.findAll { clsOf(it) == 'GT Clear cell boundary' }, false)
    drawMask(new File(auxMasks, spec.id + '_gt_compact_boundary_all.png'), boundaries.findAll { clsOf(it) == 'GT Compact cell boundary' }, false)
    drawSemantic(new File(semanticMasks, spec.id + '_semantic_review.png'), [
        stroma: [value: 4, objects: stromas],
        uncertain: [value: 3, objects: uncertain],
        clear: [value: 1, objects: boundaries.findAll { clsOf(it) == 'GT Clear cell boundary' }],
        compact: [value: 2, objects: boundaries.findAll { clsOf(it) == 'GT Compact cell boundary' }]
    ])

    File perTileCsv = new File(outDir, spec.id + '_instances.csv')
    perTileCsv.text = 'instance_label,boundary_name,boundary_class,nucleus_name,boundary_area_px,local_centroid_x,local_centroid_y,wsi_centroid_x,wsi_centroid_y,nucleus_wsi_x,nucleus_wsi_y,stroma_overlap_px,stroma_overlap_fraction\n'
    validRows.eachWithIndex { row, idx ->
        def b = row.boundary
        def n = row.nucleus
        int label = idx + 1
        double localCx = b.getROI().getCentroidX() - x
        double localCy = b.getROI().getCentroidY() - y
        def values = [label, b.getName(), clsOf(b), n.getName(), String.format('%.1f', row.area), String.format('%.1f', localCx), String.format('%.1f', localCy), String.format('%.1f', b.getROI().getCentroidX()), String.format('%.1f', b.getROI().getCentroidY()), String.format('%.1f', n.getROI().getCentroidX()), String.format('%.1f', n.getROI().getCentroidY()), String.format('%.1f', row.overlap), String.format('%.4f', row.frac)]
        perTileCsv << csvRow(values)
        instanceCsv << csvRow([spec.id, spec.name] + values)
    }

    int clearTrainable = validRows.count { clsOf(it.boundary) == 'GT Clear cell boundary' }
    int compactTrainable = validRows.count { clsOf(it.boundary) == 'GT Compact cell boundary' }
    tileCsv << csvRow([spec.id, spec.name, 'train/images/' + imageFile.getName(), 'train/masks/' + maskFile.getName(), w, h, validRows.size(), clearTrainable, compactTrainable, edgeOrInvalid.size(), uncertain.size(), nuclei.size(), nucleiInTile.size(), stromas.size()])

    // Preview overlay for visual QC.
    def preview = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    def g = preview.createGraphics()
    g.drawImage(raw, 0, 0, null)
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    g.setFont(new Font('SansSerif', Font.BOLD, 10))
    def overlayOrdered = []
    overlayOrdered.addAll(stromas)
    overlayOrdered.addAll(uncertain)
    overlayOrdered.addAll(edgeOrInvalid)
    overlayOrdered.addAll(boundaries)
    overlayOrdered.addAll(nucleiInTile)
    overlayOrdered.each { obj ->
        def geom = clipGeom(obj)
        if (geom.isEmpty()) return
        def c = colorFor(obj)
        boolean isEdgeIgnore = edgeOrInvalid.contains(obj)
        g.setColor(new Color(c.getRed(), c.getGreen(), c.getBlue(), clsOf(obj) == 'GT Stroma' ? 45 : 35))
        if (clsOf(obj) != 'GT Nucleus') g.fill(localShape(geom))
        g.setStroke(new BasicStroke(isEdgeIgnore ? 4.0f : (clsOf(obj) == 'GT Nucleus' ? 2.0f : 2.5f)))
        g.setColor(isEdgeIgnore ? Color.RED : c)
        g.draw(localShape(geom))
        if (clsOf(obj) != 'GT Stroma') {
            int lx = Math.round((float)(obj.getROI().getCentroidX() - x))
            int ly = Math.round((float)(obj.getROI().getCentroidY() - y))
            String label = obj.getName() ?: ''
            if (isEdgeIgnore) label += ' [IGNORE]'
            g.setColor(new Color(255,255,255,210))
            g.fillRect(lx + 3, ly - 12, Math.min(190, label.length() * 7 + 8), 14)
            g.setColor(isEdgeIgnore ? Color.RED : Color.BLACK)
            g.drawString(label, lx + 6, ly - 2)
        }
    }
    g.dispose()
    File previewFile = new File(previews, spec.id + '_overlay.png')
    ImageIO.write(preview, 'PNG', previewFile)

    summaryRows << [
        id: spec.id, name: spec.name, width: w, height: h, trainable: validRows.size(), clear: clearTrainable,
        compact: compactTrainable, edgeIgnore: edgeOrInvalid.size(), uncertain: uncertain.size(), nuclei: nuclei.size(),
        nucleiInTile: nucleiInTile.size(), stroma: stromas.size(), image: imageFile.getAbsolutePath(), mask: maskFile.getAbsolutePath(), preview: previewFile.getAbsolutePath()
    ]
    totalTrainable += validRows.size()
    totalClear += clearTrainable
    totalCompact += compactTrainable
    totalEdgeIgnore += edgeOrInvalid.size()
    totalUncertain += uncertain.size()
    totalNucleiInTile += nucleiInTile.size()
}

File readme = new File(outDir, 'README_export.txt')
readme.text = """CGH P2 ground-truth export for CellSeg1 and downstream morphometry models

Source QuPath project: /Volumes/T9/CGH_PA_annotation_1/project.qpproj
Source image: target.tiff
Export date: 2026-06-06
Training tiles: final GT Training tile annotations named P2 tile NN or yolo_tile_NN.
Temporary/duplicate training tile annotations such as cell_boundary_clean, nuclei_clean, and unnumbered yolo_tile are intentionally excluded.

Primary CellSeg1 files:
- train/images/*.png: one image per exported training tile
- train/masks/*.png: uint8 instance masks, 0=background, 1..N=trainable cell boundary instances

Auxiliary masks:
- auxiliary_masks/*_gt_nucleus_instances.png: in-tile GT Nucleus instance masks, excluding nuclei marked ignore or located inside GT Stroma
- auxiliary_masks/*_gt_nucleus_all_direct_children.png: direct child nuclei, including edge/outside-tile nuclei kept for review, excluding nuclei marked ignore or located inside GT Stroma
- auxiliary_masks/*_gt_stroma.png: GT Stroma binary mask, including stroma crossing the tile edge
- auxiliary_masks/*_gt_uncertain_ignore.png: GT Uncertain clear cell boundary regions for ignore/review handling
- auxiliary_masks/*_edge_or_invalid_cell_ignore.png: clear/compact cell boundaries excluded from CellSeg1 positives because the nucleus is outside the tile or the one-nucleus rule fails
- auxiliary_masks/*_gt_clear_boundary_all.png: all clear-cell boundary annotations for the tile
- auxiliary_masks/*_gt_compact_boundary_all.png: all compact-cell boundary annotations for the tile
- semantic_masks/*_semantic_review.png: review-only semantic mask; 1=clear boundary, 2=compact boundary, 3=uncertain, 4=stroma

CSV metadata:
- dataset_manifest.csv: one row per tile
- cell_instances.csv: combined label-to-annotation mapping
- *_instances.csv: per-tile instance mappings
- boundary_qc.csv: boundary inclusion flags and stroma-overlap measurements

Export rules:
- Positive CellSeg1 instances include GT Clear cell boundary and GT Compact cell boundary only.
- A positive boundary must contain exactly one GT Nucleus centroid whose centroid is inside the same training tile.
- A positive boundary is excluded if its nucleus is inside GT Stroma or if exclude_from_training_export=true.
- GT Uncertain clear cell boundary annotations are exported as ignore/review masks, not positive training instances.
- Edge cells with visible boundary but nucleus outside the tile are exported as ignore masks, not positive training instances.
- Stroma masks include GT Stroma that intersects the tile even if the stroma annotation is parented under a neighboring training tile.
- Exported stroma/uncertain/edge-ignore masks are clipped away from positive trainable instances to avoid auxiliary-mask overlap.

Current summary:
"""
summaryRows.each { row ->
    readme << String.format('- %s: %d trainable instances (%d clear, %d compact), %d edge/invalid ignored, %d uncertain ignored, %d in-tile nuclei, %d intersecting stroma regions, size %dx%d\n', row.id, row.trainable, row.clear, row.compact, row.edgeIgnore, row.uncertain, row.nucleiInTile, row.stroma, row.width, row.height)
}
readme << String.format('\nTotal trainable instances: %d (%d clear, %d compact)\nTotal uncertain ignored regions: %d\nTotal edge/invalid ignored boundaries: %d\n', totalTrainable, totalClear, totalCompact, totalUncertain, totalEdgeIgnore)

println 'EXPORT_CELLSEG1_CGH_P2_DONE'
println 'out_dir=' + outDir.getAbsolutePath()
summaryRows.each { row ->
    println String.format('TILE\t%s\ttrainable=%d\tclear=%d\tcompact=%d\tedge_ignore=%d\tuncertain_ignore=%d\tnuclei_in_tile=%d\tstroma=%d\tsize=%dx%d\tpreview=%s', row.id, row.trainable, row.clear, row.compact, row.edgeIgnore, row.uncertain, row.nucleiInTile, row.stroma, row.width, row.height, row.preview)
}
println String.format('TOTAL\ttrainable=%d\tclear=%d\tcompact=%d\tedge_ignore=%d\tuncertain_ignore=%d\tnuclei_in_tile=%d', totalTrainable, totalClear, totalCompact, totalEdgeIgnore, totalUncertain, totalNucleiInTile)
