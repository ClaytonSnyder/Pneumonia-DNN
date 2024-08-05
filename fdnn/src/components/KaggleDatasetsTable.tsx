import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";
import { KaggleDataset } from "./KaggleSearch";
import { KaggleDatasetsTableRow } from "./KaggleDatasetsTableRow";

interface KaggleDatasetsTableProps {
    datasets: Array<KaggleDataset>;
    setSelectedDatasets: (identifier: string, url: string) => void;
}

export function KaggleDatasetsTable(props: KaggleDatasetsTableProps) {
    return (
        <TableContainer sx={{ maxHeight: 300 }} component={Paper}>
            <Table stickyHeader size="small" sx={{ minWidth: 650 }} aria-label="simple table">
                <TableHead>
                    <TableRow>
                        <TableCell></TableCell>
                        <TableCell>Identifier</TableCell>
                        <TableCell>Title</TableCell>
                        <TableCell align="right">Creator</TableCell>
                        <TableCell align="right">Last Updated</TableCell>
                        <TableCell align="right">Usability</TableCell>
                        <TableCell align="right">Vote Count</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {props.datasets.map((row) => (
                        <KaggleDatasetsTableRow
                            key={row.identifier}
                            setSelectedDatasets={props.setSelectedDatasets}
                            row={row}
                        />
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
}
