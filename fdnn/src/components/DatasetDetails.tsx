import { useParams } from "react-router-dom";
import { IDataset } from "./DatasetPage";
import {
    Breadcrumbs,
    Grid,
    Link,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Typography,
    useTheme,
} from "@mui/material";
import { KaggleAdd } from "./KaggleAdd";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface IDetasetDetailsProps {
    datasets: Array<IDataset>;
    refresh: () => void;
}

interface IDataPoint {
    key: number;
    value: number;
}

interface Bin {
    key: string;
    uv: number;
}

function binData(items: Array<IDataPoint>, numBins: number): Array<Bin> {
    let minKey = Number.MAX_VALUE;
    let maxKey = Number.MIN_VALUE;

    items.forEach((item) => {
        if (item.key < minKey) minKey = item.key;
        if (item.key > maxKey) maxKey = item.key;
    });

    const range = maxKey - minKey;
    const binSize = range / numBins;

    const bins: Bin[] = new Array(numBins).fill(null).map((_, index) => ({
        key: `${Math.ceil(minKey + index * binSize)}-${Math.ceil(minKey + (index + 1) * binSize)}`,
        uv: 0,
    }));

    items.forEach((item) => {
        let index = Math.floor((item.key - minKey) / binSize);
        if (index === numBins) index = numBins - 1;
        bins[index].uv += item.value;
    });
    console.dir(bins);
    return bins;
}

function reformat_distribution(input_dist: { [key: string]: number }): Array<Bin> {
    return binData(
        Object.keys(input_dist).map((key) => {
            return {
                key: Number(key),
                value: input_dist[key],
            };
        }),
        10
    );
}

export function DetasetDetails(props: IDetasetDetailsProps) {
    let { id } = useParams();
    let dataset = props.datasets.find((h) => h.name === id);
    let theme = useTheme();
    let height_distribution = reformat_distribution(dataset ? dataset.height_distribution : {});
    let width_distribution = reformat_distribution(dataset ? dataset.width_distribution : {});

    if (dataset) {
        return (
            <Paper sx={{ margin: theme.spacing(2), padding: theme.spacing(2), textAlign: "left" }}>
                <Breadcrumbs aria-label="breadcrumb" sx={{ marginBottom: theme.spacing(1) }}>
                    <Link underline="hover" color="inherit" href="/datasets">
                        Datasets
                    </Link>
                    <Typography color="text.primary">{dataset.name}</Typography>
                </Breadcrumbs>
                <div style={{ marginTop: 30 }}>
                    <Grid container direction="row" xl={12}>
                        <Grid item xl={12}>
                            <Typography gutterBottom variant="h6">
                                Name
                            </Typography>
                            <Typography gutterBottom variant="h4">
                                {dataset.name}
                            </Typography>
                        </Grid>
                    </Grid>
                    <Grid container direction="row" xl={12}>
                        <Grid item xl={12}>
                            <Typography gutterBottom variant="h6">
                                Description
                            </Typography>
                            <Typography variant="body1">
                                {dataset.description == "" ? "None" : dataset.description}
                            </Typography>
                        </Grid>
                    </Grid>
                    <Grid container direction="row" xl={12} style={{ marginTop: 30 }}>
                        <Grid item xl={4}>
                            <Typography gutterBottom variant="h6">
                                Labels
                            </Typography>

                            {Object.entries(dataset.labels).length > 0 &&
                                Object.entries(dataset.labels).map(([key, value]) => (
                                    <Typography key={key} variant="body1">
                                        {key}: {value} images ({((value / dataset.total_image_count) * 100).toFixed(2)}
                                        %)
                                    </Typography>
                                ))}
                            {Object.entries(dataset.labels).length === 0 && (
                                <Typography variant="body1">None</Typography>
                            )}
                        </Grid>
                        <Grid item xl={2}>
                            <Typography gutterBottom variant="h6">
                                Average Height
                            </Typography>
                            <Typography gutterBottom variant="h4">
                                {dataset.avg_height}
                            </Typography>
                        </Grid>
                        <Grid item xl={2}>
                            <Typography gutterBottom variant="h6">
                                Average Width
                            </Typography>
                            <Typography gutterBottom variant="h4">
                                {dataset.avg_width}
                            </Typography>
                        </Grid>
                        <Grid item xl={2}>
                            <Typography gutterBottom variant="h6">
                                Corrupted Image Count (Removed)
                            </Typography>
                            <Typography gutterBottom variant="h4">
                                {dataset.corrupt_image_count}
                            </Typography>
                        </Grid>
                        <Grid item xl={2}>
                            <Typography gutterBottom variant="h6">
                                Total Image Count
                            </Typography>
                            <Typography gutterBottom variant="h4">
                                {dataset.total_image_count}
                            </Typography>
                        </Grid>
                    </Grid>
                    <Grid container direction="row" xl={12} sx={{ mt: 5, height: 250 }}>
                        <Grid item xl={6}>
                            <Typography gutterBottom variant="h6">
                                Width Distribution
                            </Typography>
                            {width_distribution.length > 0 ? (
                                <ResponsiveContainer width="100%" height="80%">
                                    <LineChart
                                        data={width_distribution}
                                        margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
                                    >
                                        <Line type="monotone" dataKey="uv" stroke="#8884d8" />
                                        <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                                        <XAxis dataKey="key" />
                                        <YAxis />
                                        <Tooltip />
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <Typography variant="body1">None</Typography>
                            )}
                        </Grid>
                        <Grid item xl={6}>
                            <Typography gutterBottom variant="h6">
                                Height Distribution
                            </Typography>
                            {height_distribution.length > 0 ? (
                                <ResponsiveContainer width="100%" height="80%">
                                    <LineChart
                                        data={height_distribution}
                                        margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
                                    >
                                        <Line type="monotone" dataKey="uv" stroke="#8884d8" />
                                        <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                                        <XAxis dataKey="key" />
                                        <YAxis />
                                        <Tooltip />
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <Typography variant="body1">None</Typography>
                            )}
                        </Grid>
                    </Grid>
                    <div style={{ marginTop: theme.spacing(3) }}>
                        <Typography gutterBottom variant="h6">
                            Kaggle Data Sources
                        </Typography>
                        <KaggleAdd dataset={dataset} refresh={props.refresh} />

                        <TableContainer sx={{ mt: 4 }} component={Paper}>
                            <Table stickyHeader size="small" sx={{ minWidth: 650 }} aria-label="simple table">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Identifier</TableCell>
                                        <TableCell>Url</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {dataset.kaggle_datasets_included.map((row) => (
                                        <TableRow
                                            key={row.identifier}
                                            sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
                                        >
                                            <TableCell component="th" scope="row">
                                                {row.identifier}
                                            </TableCell>
                                            <TableCell component="th" scope="row">
                                                <Link href={row.url}>{row.url}</Link>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </div>
                </div>
            </Paper>
        );
    } else {
        return (
            <div>
                <h3>ID: {id}</h3>
            </div>
        );
    }
}
